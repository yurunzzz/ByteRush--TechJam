"""Backfill canonical dashboard snapshots for historic ByteRush experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_scientist.treesearch.utils.dashboard_snapshot import rebuild_dashboard_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    runs = sorted(path for path in args.experiments.iterdir() if path.is_dir())
    written = []
    for run in runs:
        snapshot = rebuild_dashboard_snapshot(run)
        if snapshot:
            written.append(snapshot)
    print(f"Wrote {len(written)} dashboard snapshots under {args.experiments}")


if __name__ == "__main__":
    main()
