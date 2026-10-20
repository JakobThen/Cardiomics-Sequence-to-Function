"""Prints one resolved path from config/paths.yaml (used by `cs2f_path` in shell scripts).

Usage: python get_path.py <key>

Loads ``cardiomics_s2f.config`` straight from the repository so that it also works
in environments where the package is not installed (only PyYAML is required).
"""
import importlib.util
import sys
from pathlib import Path

CONFIG_MODULE = Path(__file__).resolve().parents[2] / "src" / "cardiomics_s2f" / "config.py"


def load_config_module():
    spec = importlib.util.spec_from_file_location("cs2f_config", CONFIG_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv):
    if len(argv) != 2:
        print("Usage: get_path.py <key>", file=sys.stderr)
        return 2
    try:
        print(load_config_module().load_paths()[argv[1]].as_posix())
    except (KeyError, FileNotFoundError, ValueError) as err:
        print(f"Error: {err.args[0] if err.args else err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
