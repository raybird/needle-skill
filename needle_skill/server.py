import json
import logging
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import engine, config as cfgmod

logger = logging.getLogger("needle-skill")

_STATUS = {"model": "", "params": 0, "uptime": 0, "requests": 0}


class NeedleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/health":
            self._json(200, {
                "ok": True,
                "model": _STATUS["model"],
                "params": _STATUS["params"],
                "uptime": int(__import__("time").time() - _STATUS["uptime"]),
                "requests": _STATUS["requests"],
            })
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path == "/v1/call":
            self._handle_call()
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def _handle_call(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"ok": False, "error": "invalid JSON"})
            return

        query = body.get("query")
        tools = body.get("tools", "[]")
        max_gen_len = body.get("max_gen_len", cfgmod.ensure_config()["model"]["max_gen_len"])
        constrained = body.get("constrained", cfgmod.ensure_config()["model"]["constrained"])
        seed = body.get("seed", 0)

        if not query or not isinstance(query, str):
            self._json(400, {"ok": False, "error": "query is required"})
            return

        try:
            result = engine.generate(query, tools, max_gen_len=max_gen_len,
                                     seed=seed, constrained=constrained)
            result = result.strip()
            _STATUS["requests"] += 1
            self._json(200, {"ok": True, "result": result})
        except Exception as exc:
            logger.exception("generate failed")
            self._json(500, {"ok": False, "error": str(exc)})

    def _json(self, code, data):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def run_server(host, port, checkpoint_path):
    if engine.is_loaded():
        engine.unload()
    engine.load_model(checkpoint_path)

    import time
    import jax

    _STATUS["model"] = Path(checkpoint_path).name
    _STATUS["params"] = sum(x.size for x in jax.tree.leaves(engine._PARAMS))
    _STATUS["uptime"] = time.time()
    _STATUS["requests"] = 0

    server = ThreadingHTTPServer((host, port), NeedleHandler)
    server.timeout = 30

    pid = os.getpid()
    cfgmod.pid_file().write_text(str(pid))

    print(f"Needle server running at http://{host}:{port}", file=sys.stderr)
    print(f"PID: {pid}", file=sys.stderr)

    def signal_handler(sig, frame):
        print("\nShutting down...", file=sys.stderr)
        server.shutdown()
        cfgmod.pid_file().unlink(missing_ok=True)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cfgmod.pid_file().unlink(missing_ok=True)
        print("Server stopped.", file=sys.stderr)