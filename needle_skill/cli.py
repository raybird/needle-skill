import argparse
import json
import os
import signal
import socket
import sys
from pathlib import Path

from . import config as cfgmod


def _port_available(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def cmd_setup(args):
    cfgmod.setup_needledir()


def cmd_serve(args):
    host = args.host
    port = args.port

    if not _port_available(host, port):
        print(f"PORT {port} already in use. Use --port <port> to specify a different port.",
              file=sys.stderr)
        sys.exit(1)

    cfg = cfgmod.ensure_config()
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = cfgmod.resolve_checkpoint(cfg)

    if not Path(checkpoint).exists():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        print("Run 'needle-skill setup' to download the checkpoint.", file=sys.stderr)
        sys.exit(1)

    from .server import run_server
    run_server(host, port, checkpoint)


def cmd_stop(args):
    pid_path = cfgmod.pid_file()
    if not pid_path.exists():
        print("No running server found.", file=sys.stderr)
        return
    try:
        pid = int(pid_path.read_text().strip())
        print(f"Stopping server (PID {pid})...", file=sys.stderr)
        os.kill(pid, signal.SIGTERM)
        pid_path.unlink(missing_ok=True)
    except ProcessLookupError:
        print("Server process not found. Removing stale PID file.", file=sys.stderr)
        pid_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"Failed to stop server: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    pid_path = cfgmod.pid_file()
    if not pid_path.exists():
        print(json.dumps({"running": False}))
        return
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        print(json.dumps({"running": True, "pid": pid}))
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        print(json.dumps({"running": False}))
    except Exception as exc:
        print(json.dumps({"running": False, "error": str(exc)}))


def cmd_call(args):
    import urllib.request

    cfg = cfgmod.ensure_config()
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]

    tools = args.tools
    if args.tools_file:
        tools = Path(args.tools_file).read_text().strip()

    body = json.dumps({
        "query": args.query,
        "tools": tools,
        "max_gen_len": args.max_gen_len or cfg["model"]["max_gen_len"],
        "constrained": args.constrained if args.constrained is not None else cfg["model"]["constrained"],
        "seed": args.seed,
    }).encode()

    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/call",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            result = json.loads(resp.read())
            print(json.dumps(result, ensure_ascii=False))
            if not result.get("ok"):
                sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Failed to connect to server at {host}:{port}", file=sys.stderr)
        print("Is the server running? Run 'needle-skill serve' first.", file=sys.stderr)
        sys.exit(1)


def cmd_health(args):
    import urllib.request

    cfg = cfgmod.ensure_config()
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]

    try:
        req = urllib.request.Request(f"http://{host}:{port}/v1/health")
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            print(resp.read().decode())
    except urllib.error.URLError:
        print(json.dumps({"ok": False, "error": "server unreachable"}), file=sys.stderr)
        sys.exit(1)


def cmd_skill(args):
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        print(f"Workspace not found: {workspace}", file=sys.stderr)
        sys.exit(1)

    skill_src = Path(__file__).parent / "SKILL.md"
    if not skill_src.exists():
        print("Skill file not found in package.", file=sys.stderr)
        sys.exit(1)

    target_dir = workspace / ".opencode" / "skills" / "needle-skill"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "SKILL.md"

    if args.copy:
        target_file.write_text(skill_src.read_text())
        print(f"Skill copied to {target_file}")
    else:
        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()
        os.symlink(str(skill_src), str(target_file))
        print(f"Skill symlinked: {target_file} -> {skill_src}")


def parse(args=None):
    parser = argparse.ArgumentParser(prog="needle-skill")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("setup", help="Initialize ~/.needle/, install Needle, download checkpoint")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("serve", help="Start HTTP server")
    p.add_argument("--host", default=None, help="Server host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="Server port (default: 3918)")
    p.add_argument("--checkpoint", default=None, help="Path to checkpoint.pkl")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("stop", help="Stop running server")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("status", help="Show server status (JSON)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("call", help="Send a function calling request to server")
    p.add_argument("--query", required=True, help="Query text")
    p.add_argument("--tools", default=None, help="Tools JSON string")
    p.add_argument("--tools-file", default=None, help="Read tools from file")
    p.add_argument("--host", default=None, help="Server host")
    p.add_argument("--port", type=int, default=None, help="Server port")
    p.add_argument("--max-gen-len", type=int, default=None, help="Max generation length")
    p.add_argument("--constrained", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    p.set_defaults(func=cmd_call)

    p = sub.add_parser("health", help="Check server health")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--timeout", type=int, default=10)
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("skill", help="Install skill into an opencode workspace")
    p.add_argument("workspace", help="Path to opencode workspace directory")
    p.add_argument("--copy", action="store_true", help="Copy file instead of symlink")
    p.set_defaults(func=cmd_skill)

    return parser.parse_args(args)


def main(args=None):
    ns = parse(args)
    if not ns.command:
        parse(["-h"])
        return

    if ns.command not in ("setup", "stop", "status", "skill"):
        host = getattr(ns, "host", None)
        port = getattr(ns, "port", None)
        if host is None or port is None:
            cfg = cfgmod.ensure_config()
            if host is None:
                ns.host = cfg["server"]["host"]
            if port is None:
                ns.port = cfg["server"]["port"]

    ns.func(ns)