"""Build trusted raw-data and factor context for each KuaiRand research round."""
from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


def _dataset_dir(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    candidates = (
        root / "KuaiRand-Pure" / "data",
        root / "data",
        root,
    )
    for candidate in candidates:
        if (candidate / "log_standard_4_08_to_4_21_pure.csv").is_file():
            return candidate
    raise FileNotFoundError(f"KuaiRand-Pure data directory not found under {root}")


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return list(next(csv.reader(handle)))


def _literal_mapping(code: str, variable: str) -> Mapping[str, Any]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return {}
    for statement in tree.body:
        target = None
        value = None
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(item, ast.Name) and item.id == variable
                for item in statement.targets
            ):
                target, value = variable, statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == variable
        ):
            target, value = variable, statement.value
        if target is not None:
            try:
                result = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                return {}
            return result if isinstance(result, Mapping) else {}
    return {}


def _literal_sequence(code: str, variable: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return []
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(item, ast.Name) and item.id == variable
            for item in statement.targets
        ):
            continue
        try:
            result = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError):
            return []
        if isinstance(result, (list, tuple)):
            return [str(item) for item in result]
    return []


def _unused_factor_groups(code: str) -> list[tuple[str, str]]:
    lowered = code.lower()
    groups = [
        (
            "temporal_context",
            "hour-of-day, weekday, recency, and log-duration buckets from "
            "date/hourmin/time_ms/duration_ms",
        ),
        (
            "static_user_profile",
            "user activity degree, creator/live status, registration age, and "
            "follow/fan/friend ranges with explicit missing-value buckets",
        ),
        (
            "static_video_metadata",
            "video type, upload age/type, aspect ratio, music, and tag metadata",
        ),
        (
            "causal_user_history",
            "past exposure counts and recency from prediction-time-known fields; "
            "outcome-based rates/watch summaries must be frozen from train only",
        ),
        (
            "causal_affinity",
            "past-only user-author, user-tag, and user-video-type affinity factors",
        ),
        (
            "auxiliary_training_targets",
            "click/like/follow/comment/forward and watch time used only as "
            "train-window targets or train-window history, never from valid/test",
        ),
    ]
    return [
        group
        for group in groups
        if group[0] not in lowered and group[0].replace("_", " ") not in lowered
    ]


def build_factor_context(
    data_dir: str | Path,
    *,
    incumbent_code: str,
    round_number: int,
    failed_hypotheses: Iterable[str] = (),
) -> str:
    """Return a deterministic, leakage-aware context appended to Stage 3."""
    dataset_dir = _dataset_dir(data_dir)
    files = {
        "interaction_log": "log_standard_4_08_to_4_21_pure.csv",
        "user_profile": "user_features_pure.csv",
        "video_basic": "video_features_basic_pure.csv",
        "video_statistics": "video_features_statistic_pure.csv",
    }
    headers = {
        name: _csv_header(dataset_dir / filename)
        for name, filename in files.items()
    }
    components = sorted(
        name
        for name, enabled in _literal_mapping(
            incumbent_code, "ABLATION_COMPONENTS"
        ).items()
        if enabled is True
    )
    encoded_fields = _literal_sequence(incumbent_code, "FIELDS")
    factor_groups = _unused_factor_groups(incumbent_code)
    rotation = (max(1, round_number) - 1) % max(1, len(factor_groups))
    prioritized = factor_groups[rotation:] + factor_groups[:rotation]
    failed = list(failed_hypotheses)[-6:]

    lines = [
        "Trusted KuaiRand raw-data/factor context for this research round:",
        f"- Round: {round_number}",
        "- Permitted feedback: fixed train and validation only; test is prediction-only.",
        f"- Interaction columns: {', '.join(headers['interaction_log'])}",
        f"- User profile columns: {', '.join(headers['user_profile'])}",
        f"- Basic video columns: {', '.join(headers['video_basic'])}",
        f"- Current literal FIELDS: {', '.join(encoded_fields) or 'defined inside the candidate builder'}",
        f"- Current registered components: {', '.join(components) or 'none'}",
        "- Exact runtime split type: dict[str, list[tuple]]. It is not a pandas DataFrame and has no sort_values/groupby methods.",
        "- Each official raw tuple is exactly (date:int, user_id:str, video_id:str, author_id:str, tab:str, duration_ms:float, long_view:int).",
        "- Tuple indices are 0=date, 1=user_id, 2=video_id, 3=author_id, 4=tab, 5=duration_ms, 6=long_view label.",
        "- CSV columns not listed in that tuple, including time_ms, are not automatically present in splits. Do not assume header names are row attributes.",
        "- For vectorized indexing, explicitly convert a split list with numpy or use list comprehensions; never apply a numpy index array directly to a Python list.",
        "- build_features must preserve the incumbent return contract: encoded split -> (X int32 shape (N,F), y float32 shape (N,), users list), feature_dimension int, feature_state mapping.",
        "- Prioritized unused factor directions:",
    ]
    lines.extend(f"  {index}. {name}: {description}" for index, (name, description) in enumerate(prioritized[:4], 1))
    if failed:
        lines.append("- Recent rejected hypotheses; do not repeat unchanged:")
        lines.extend(f"  - {reason}" for reason in failed)
    lines.extend(
        [
            "- Mandatory feature contract:",
            "  1. Fit vocabularies, bins, scalers, and aggregate priors on train only.",
            "  2. Freeze every label/outcome-derived history at the end of train. Never read validation/test outcomes, even from rows with an earlier time_ms.",
            "  3. Validation/test may update only non-outcome context known at prediction time, and only from earlier time_ms rows.",
            "  4. Use the same frozen feature builder and field order for train/valid/test.",
            "  5. Current-row long_view, click/like/follow/comment/forward, play_time, and stay-time outcomes are forbidden inputs.",
            "  6. Treat video_features_statistic_pure.csv as unsafe unless every statistic is rebuilt from the train window with an explicit cutoff.",
            "  7. The random-exposure log is diagnostic-only and cannot replace validation or drive promotion.",
            "  8. Register each new model/factor block as a literal True ABLATION_COMPONENTS entry.",
            "- Propose one principal algorithm/factor change in each sibling branch.",
        ]
    )
    return "\n".join(lines)
