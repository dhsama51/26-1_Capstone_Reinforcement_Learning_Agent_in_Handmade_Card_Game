#!/usr/bin/env python3
"""Windows launcher for TCP 8-combo balance evaluation."""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def _format_elapsed(seconds: float) -> str:
    seconds_i = int(max(0, seconds))
    hours, remainder = divmod(seconds_i, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d} ({seconds:.1f}s)"


def _configure_windows_env() -> None:
    home = Path.home().resolve()
    os.chdir(str(home))
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


def main() -> int:
    started_at = time.perf_counter()
    print(f"[*] make_balance_tcp_w.py start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        _configure_windows_env()
        import RL_AI.make_balance_tcp as tcp_module
        return int(tcp_module.main())

    except Exception as exc:
        print("[!] make_balance_tcp_w.py failed")
        print(f"[!] error: {exc}")
        print(traceback.format_exc())
        return 1

    finally:
        print(f"[*] make_balance_tcp_w.py total runtime: {_format_elapsed(time.perf_counter() - started_at)}")


if __name__ == "__main__":
    raise SystemExit(main())
