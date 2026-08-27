#!/usr/bin/env python3
"""Verify AI Scientist-v2's execution engine against the trusted FM harness."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ai_scientist.treesearch.interpreter import Interpreter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("deployment_runs/v2_fm_baseline"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    starter = repo / "kuairand-starter-kit"
    code_path = repo / "ai_scientist" / "ideas" / "kuairand_ranking.py"
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "working").mkdir(exist_ok=True)
    input_link = workspace / "input"
    if input_link.is_symlink() and input_link.resolve() != starter.resolve():
        input_link.unlink()
    if not input_link.exists():
        input_link.symlink_to(starter.resolve(), target_is_directory=True)

    interpreter = Interpreter(
        working_dir=workspace,
        timeout=600,
        agent_file_name="runfile.py",
    )
    try:
        execution = interpreter.run(code_path.read_text(encoding="utf-8"), reset_session=True)
    finally:
        interpreter.cleanup_session()

    output = "".join(execution.term_out)
    print(output, end="")
    if execution.exc_type is not None:
        raise RuntimeError(f"V2 interpreter failed: {execution.exc_type}: {execution.exc_info}")

    artifact = workspace / "working" / "experiment_data.npy"
    if not artifact.exists():
        raise RuntimeError("V2 run did not create experiment_data.npy")
    experiment_data = np.load(artifact, allow_pickle=True).item()
    metrics = experiment_data["KuaiRand-Pure"]["metrics"]
    primary = float(metrics["validation primary"][-1])
    if not 0.59 <= primary <= 0.61:
        raise RuntimeError(f"FM baseline primary outside expected range: {primary}")

    report = {
        "status": "success",
        "execution_engine": "ai_scientist.treesearch.interpreter.Interpreter",
        "split": "valid",
        "primary": primary,
        "workspace": str(workspace),
        "experiment_data": str(artifact),
    }
    report_path = workspace / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("V2_FM_SMOKE_RESULT " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
