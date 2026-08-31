#!/usr/bin/env python3
"""Backfill explicit iteration code-diff logs for one experiment run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_scientist.treesearch.utils.code_diff import write_run_logs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="experiment directory name or path")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        direct = (ROOT / run).resolve()
        under_experiments = (ROOT / "experiments" / run.name).resolve()
        run = direct if direct.is_dir() else under_experiments
    if not run.is_dir():
        raise SystemExit(f"run directory not found: {run}")
    json_path, markdown_path = write_run_logs(run)
    document_size = json_path.stat().st_size
    print(f"[code_diff] wrote {json_path.relative_to(ROOT)} ({document_size} bytes)")
    print(f"[code_diff] wrote {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
