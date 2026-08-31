"""Generate the frozen data contract consumed by the ByteRush showcase UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from showcase_loader import ShowcaseBuildError, build_showcase_payload, write_showcase_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen ByteRush competition showcase manifest.")
    parser.add_argument("--data-root", type=Path, required=True, help="ByteRush directory containing experiments/ and artifacts/.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "showcase_config.yaml")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "generated" / "showcase_manifest.json")
    args = parser.parse_args()
    try:
        payload = build_showcase_payload(args.data_root, args.config)
    except ShowcaseBuildError as exc:
        raise SystemExit(f"Showcase build refused: {exc}") from exc
    output = write_showcase_payload(payload, args.output)
    winner = payload["winner"]
    summary = {
        "output": str(output.resolve()),
        "run_id": payload["selection"]["run_id"],
        "winner": winner["label"],
        "primary": winner["final"]["primary"],
        "primary_delta": winner["delta"]["primary"],
        "seeds": payload["selection"]["successful_seed_count"],
        "submission_rows": payload["integrity"]["submission_rows"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

