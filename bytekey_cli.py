#!/usr/bin/env python3
"""Thin entry point so the CLI runs as ``python bytekey_cli.py <command>``.

It ensures the package directory is importable even under ``python -I``
(isolated mode), where the current directory and PYTHONPATH are ignored.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bytekey.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
