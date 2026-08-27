import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

execution_dir = Path.cwd()
input_candidates = (execution_dir / "input", execution_dir.parent / "input")
input_dir = next((path for path in input_candidates if path.is_dir()), None)
if input_dir is None:
    searched = ", ".join(str(path) for path in input_candidates)
    raise FileNotFoundError(f"Could not locate the trusted input directory; searched: {searched}")
working_dir = execution_dir / "working"
working_dir.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable,
    str(input_dir / "run_fm_experiment.py"),
    "--config",
    str(input_dir / "fm_experiment_config.json"),
    "--data-dir",
    str(input_dir / "KuaiRand-Pure" / "data"),
    "--output-dir",
    str(working_dir / "fm_validation_baseline"),
]
completed = subprocess.run(
    command,
    cwd=str(input_dir),
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
)
print(completed.stdout, end="")
if completed.returncode != 0:
    raise RuntimeError(f"trusted FM harness exited with code {completed.returncode}")

prefix = "AI_SCIENTIST_RESULT "
result_lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
if len(result_lines) != 1:
    raise RuntimeError("trusted FM harness did not emit exactly one result record")
result = json.loads(result_lines[0][len(prefix) :])
if result.get("status") != "success" or result.get("split") != "valid":
    raise RuntimeError(f"invalid trusted result: {result}")
if any(key.lower().startswith("test") for key in result):
    raise RuntimeError("test feedback appeared in the agent-visible result")

gauc = float(result["GAUC"])
ndcg5 = float(result["nDCG@5"])
primary = float(result["primary"])
if not np.isfinite([gauc, ndcg5, primary]).all():
    raise RuntimeError("validation metrics contain NaN or infinity")
if abs(primary - (gauc + ndcg5) / 2.0) > 1e-12:
    raise RuntimeError("primary does not equal the official metric mean")

experiment_data = {
    "KuaiRand-Pure": {
        "metrics": {
            "validation primary": [primary],
            "validation GAUC": [gauc],
            "validation nDCG@5": [ndcg5],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "metadata": {
            "metric_to_optimize": "validation primary",
            "maximize": True,
            "best_epoch": int(result["best_epoch"]),
            "seed": int(result["seed"]),
            "checkpoint": result["checkpoint"],
        },
    }
}
np.save(working_dir / "experiment_data.npy", experiment_data)
print(f"validation GAUC: {gauc:.9f}")
print(f"validation nDCG@5: {ndcg5:.9f}")
print(f"validation primary: {primary:.9f}")
