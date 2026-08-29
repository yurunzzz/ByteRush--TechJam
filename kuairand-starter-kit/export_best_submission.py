"""Generate a test submission from the Stage-4 frozen validation winner.

This command never evaluates test labels and never returns test metrics to the
agent. It loads the exact Stage-4 checkpoint selected on validation and uses it
once to score the official test rows. No retraining is allowed during export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_submission(
    output: str | Path,
    artifact_dir: str | Path,
    starter_kit: str | Path,
    seed: int | None = None,
) -> dict:
    output = Path(output).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    starter_kit = Path(starter_kit).resolve()
    model_path = artifact_dir / "model.py"
    manifest_path = artifact_dir / "manifest.json"
    checkpoint_path = artifact_dir / "checkpoint.npz"
    if (
        not model_path.is_file()
        or not manifest_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise FileNotFoundError(
            "frozen artifact must contain model.py, manifest.json, and "
            f"checkpoint.npz: {artifact_dir}"
        )
    manifest = json.loads(manifest_path.read_text())
    actual_model_hash = _sha256(model_path)
    actual_checkpoint_hash = _sha256(checkpoint_path)
    if actual_model_hash != manifest.get("model_sha256"):
        raise RuntimeError("frozen model.py hash does not match manifest.json")
    if actual_checkpoint_hash != manifest.get("checkpoint_sha256"):
        raise RuntimeError("frozen checkpoint hash does not match manifest.json")
    data_dir = starter_kit / "KuaiRand-Pure" / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"KuaiRand-Pure data directory not found: {data_dir}")

    old_target = os.environ.get("AI_SCIENTIST_ABLATION_TARGET")
    old_inference_only = os.environ.get("AI_SCIENTIST_INFERENCE_ONLY")
    os.environ["AI_SCIENTIST_ABLATION_TARGET"] = str(
        manifest.get("ablation_target", "full")
    )
    os.environ["AI_SCIENTIST_INFERENCE_ONLY"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="kuairand_final_export_") as tmp:
            workdir = Path(tmp)
            (workdir / "input").symlink_to(starter_kit, target_is_directory=True)
            with _working_directory(workdir):
                namespace = runpy.run_path(str(model_path), run_name="__final_model__")
            data_module = namespace.get("data_module")
            build_features = namespace.get("build_features")
            load_checkpoint = namespace.get("load_candidate_checkpoint")
            if (
                data_module is None
                or build_features is None
                or load_checkpoint is None
            ):
                raise RuntimeError(
                    "frozen candidate does not expose data_module, build_features, "
                    "and load_candidate_checkpoint"
                )
            frozen = load_checkpoint(checkpoint_path)
            model = frozen["model"]
            checkpoint_seed = int(frozen["config"]["seed"])
            if seed is not None and int(seed) != checkpoint_seed:
                raise ValueError(
                    f"requested seed {seed} does not match frozen checkpoint "
                    f"seed {checkpoint_seed}; export cannot retrain"
                )
            loaded = data_module.load(str(data_dir))
            encoded, feature_dimension, _ = build_features(
                {"test": loaded["test"]},
                feature_state=frozen["feature_state"],
            )
            if feature_dimension != int(frozen["feature_dimension"]):
                raise RuntimeError(
                    "frozen feature dimension does not match checkpoint model"
                )
            test_x, _, _ = encoded["test"]
            scores = np.asarray(model.predict(test_x), dtype=np.float64)
            if scores.shape != (len(loaded["test"]),):
                raise RuntimeError(
                    f"test prediction shape {scores.shape}, expected ({len(loaded['test'])},)"
                )
            if not np.isfinite(scores).all():
                raise RuntimeError("test predictions contain NaN or infinity")

            # Import after the frozen program has restored cwd. This module only
            # writes/checks schema; it does not evaluate the test labels.
            import submit as submit_module

            output.parent.mkdir(parents=True, exist_ok=True)
            submit_module.write_submission(output, loaded["test"], scores)
            submit_module.read_submission(output, loaded["test"])
    finally:
        if old_target is None:
            os.environ.pop("AI_SCIENTIST_ABLATION_TARGET", None)
        else:
            os.environ["AI_SCIENTIST_ABLATION_TARGET"] = old_target
        if old_inference_only is None:
            os.environ.pop("AI_SCIENTIST_INFERENCE_ONLY", None)
        else:
            os.environ["AI_SCIENTIST_INFERENCE_ONLY"] = old_inference_only

    metadata = {
        "artifact_dir": str(artifact_dir),
        "source_node_id": manifest.get("source_node_id"),
        "model_sha256": actual_model_hash,
        "checkpoint_sha256": actual_checkpoint_hash,
        "seed": checkpoint_seed,
        "split": "test",
        "rows": int(len(scores)),
        "submission": str(output),
        "submission_sha256": _sha256(output),
        "validation_result_from_checkpoint": frozen["metadata"].get("validation"),
        "test_metrics_computed": False,
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output} ({len(scores):,d} test rows)")
    print(f"Schema/alignment check passed; metadata: {metadata_path}")
    print("No test metric was computed or exposed.")
    return metadata


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact_dir", default=str(here.parent / "artifacts" / "final_model"))
    parser.add_argument("--starter_kit", default=str(here))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional assertion against the frozen checkpoint seed",
    )
    args = parser.parse_args()
    export_submission(args.output, args.artifact_dir, args.starter_kit, args.seed)


if __name__ == "__main__":
    main()
