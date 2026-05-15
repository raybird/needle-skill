import os
import sys
import yaml
from pathlib import Path


def needledir():
    return Path(os.path.expanduser("~/.needle"))


def config_path():
    return needledir() / "config.yaml"


def cache_dir():
    return needledir() / "cache"


def logs_dir():
    return needledir() / "logs"


def pid_file():
    return needledir() / "server.pid"


def default_checkpoint():
    return cache_dir() / "needle.pkl"


DEFAULT_CONFIG = {
    "server": {
        "host": "127.0.0.1",
        "port": 3918,
    },
    "model": {
        "checkpoint": str(default_checkpoint()),
        "max_gen_len": 512,
        "constrained": False,
    },
}


def load_config():
    cp = config_path()
    if cp.exists():
        with open(cp) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg):
    needledir().mkdir(parents=True, exist_ok=True)
    with open(config_path(), "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)


def ensure_config():
    cp = config_path()
    if not cp.exists():
        save_config(DEFAULT_CONFIG)
    cfg = load_config()
    cfg["model"]["checkpoint"] = os.path.expanduser(cfg["model"]["checkpoint"])
    return cfg


def resolve_checkpoint(cfg):
    path = Path(cfg["model"]["checkpoint"])
    if path.exists():
        return str(path)
    alt = default_checkpoint()
    if alt.exists():
        return str(alt)
    return str(path)


def is_needle_installed():
    try:
        import needle
        return True
    except ImportError:
        return False


def install_needle():
    import subprocess
    print("Installing Needle from GitHub...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/cactus-compute/needle.git"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Failed to install Needle:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


def download_checkpoint():
    import subprocess
    path = str(default_checkpoint())
    if default_checkpoint().exists():
        print(f"Checkpoint already exists at {path}")
        return path

    print("Downloading checkpoint from HuggingFace...")
    result = subprocess.run(
        [sys.executable, "-c", f"""
from huggingface_hub import hf_hub_download
hf_hub_download("Cactus-Compute/needle", "needle.pkl", local_dir="{cache_dir()}")
        """],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Failed to download checkpoint:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    print(f"Checkpoint saved to {path}")
    return path


def setup_needledir():
    for d in [needledir(), cache_dir(), logs_dir()]:
        d.mkdir(parents=True, exist_ok=True)

    if not is_needle_installed():
        install_needle()

    download_checkpoint()
    ensure_config()
    print("Setup complete.")