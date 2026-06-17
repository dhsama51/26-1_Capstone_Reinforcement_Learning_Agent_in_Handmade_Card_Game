#!/usr/bin/env python3
"""Shim that forwards to the repository-root make_balance_four.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    root_script = Path(__file__).resolve().parent.parent / "make_balance_four.py"
    spec = importlib.util.spec_from_file_location("make_balance_four_root", root_script)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load root script: {root_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError(f"root script has no main(): {root_script}")
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
