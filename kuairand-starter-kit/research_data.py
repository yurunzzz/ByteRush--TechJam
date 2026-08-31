"""Trusted schema-v2 research interface for KuaiRand candidates.

The organizer ``data.py`` remains the source of split membership, labels, and
categorical encoding.  This module only exposes that result through a richer,
explicit contract and supplies leakage-safe builders needed by ranking,
history, and multi-task candidates.
"""
from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
TRAIN_AUXILIARY_FIELDS = frozenset(
    {
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "play_time_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
    }
)


def build_schema_v2(splits, *, data_module, feature_state=None):
    """Wrap the official encoder without changing any encoded FM value."""
    encoded, feature_dimension, fitted_state = data_module.encode(
        splits,
        feature_state=feature_state,
        return_state=True,
    )
    split_payloads = {}
    for split_name, (x, y, users) in encoded.items():
        categorical_ids = np.asarray(x, dtype=np.int32)
        long_view = np.asarray(y, dtype=np.float32)
        if categorical_ids.ndim != 2 or len(categorical_ids) != len(long_view):
            raise ValueError(f"invalid official encoding for split {split_name!r}")
        split_payloads[split_name] = {
            "inputs": {
                "categorical_ids": categorical_ids,
                "user_ids": categorical_ids[:, 0],
                "item_ids": categorical_ids[:, 1],
            },
            "targets": {"long_view": long_view},
            "users": users if isinstance(users, np.ndarray) else list(users),
            "row_indices": np.arange(len(long_view), dtype=np.int64),
        }
        history_end = getattr(splits[split_name], "history_end", None)
        if history_end is not None:
            split_payloads[split_name]["inputs"]["history_end"] = np.asarray(
                history_end, dtype=np.int32
            )
    schema = {
        "schema_version": SCHEMA_VERSION,
        "splits": split_payloads,
        "feature_dimension": int(feature_dimension),
        "protocol": {
            "selection_split": "validation",
            "primary_target": "long_view",
            "test_metrics_visible_to_agent": False,
            "fitted_state_scope": "train_only",
        },
    }
    return schema, int(feature_dimension), fitted_state


class LegacyFMAdapter:
    """Lossless adapter from schema v2 to the original FM tuple contract."""

    @staticmethod
    def to_legacy(schema):
        if int(schema.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("LegacyFMAdapter requires research schema v2")
        return {
            split_name: (
                payload["inputs"]["categorical_ids"],
                payload["targets"]["long_view"],
                payload["users"],
            )
            for split_name, payload in schema["splits"].items()
        }


def assert_legacy_equivalence(official_encoded, adapted_encoded):
    """Fail fast if the compatibility layer changes any FM-visible value."""
    if set(official_encoded) != set(adapted_encoded):
        raise AssertionError("legacy adapter changed split names")
    for split_name in official_encoded:
        old_x, old_y, old_users = official_encoded[split_name]
        new_x, new_y, new_users = adapted_encoded[split_name]
        if not np.array_equal(old_x, new_x):
            raise AssertionError(f"legacy adapter changed {split_name} X")
        if not np.array_equal(old_y, new_y):
            raise AssertionError(f"legacy adapter changed {split_name} y")
        if list(old_users) != list(new_users):
            raise AssertionError(f"legacy adapter changed {split_name} users")


def same_user_pair_indices(
    users: Sequence,
    labels: Sequence,
    *,
    seed: int = 0,
    max_pairs_per_user: int = 8,
):
    """Create train-only positive/negative pairs, never cross-user pairs."""
    labels = np.asarray(labels, dtype=np.float32)
    if len(users) != len(labels):
        raise ValueError("users and labels must have equal length")
    grouped = defaultdict(lambda: {"positive": [], "negative": []})
    for index, (user, label) in enumerate(zip(users, labels)):
        grouped[str(user)]["positive" if label > 0 else "negative"].append(index)
    rng = np.random.default_rng(int(seed))
    positive_indices = []
    negative_indices = []
    for user in sorted(grouped):
        positives = grouped[user]["positive"]
        negatives = grouped[user]["negative"]
        if not positives or not negatives:
            continue
        pair_count = min(
            max(1, int(max_pairs_per_user)),
            max(len(positives), len(negatives)),
        )
        positive_indices.extend(rng.choice(positives, pair_count, replace=True))
        negative_indices.extend(rng.choice(negatives, pair_count, replace=True))
    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
    )


def attach_causal_history(schema, raw_splits, *, max_length: int = 50):
    """Attach past-only encoded item/author histories.

    Train rows see only earlier train exposures.  Validation and test receive a
    frozen end-of-train history; their outcomes never update feature state.
    """
    max_length = max(1, int(max_length))
    history_by_user = defaultdict(lambda: {"item": deque(maxlen=max_length), "author": deque(maxlen=max_length)})
    for split_name in schema["splits"]:
        payload = schema["splits"][split_name]
        categorical_ids = payload["inputs"]["categorical_ids"]
        raw_rows = raw_splits[split_name]
        if len(raw_rows) != len(categorical_ids):
            raise ValueError(f"raw/schema row mismatch for {split_name}")
        item_history = np.zeros((len(raw_rows), max_length), dtype=np.int32)
        author_history = np.zeros((len(raw_rows), max_length), dtype=np.int32)
        history_mask = np.zeros((len(raw_rows), max_length), dtype=np.bool_)
        for row_index, row in enumerate(raw_rows):
            user = str(row[1])
            state = history_by_user[user]
            item_values = list(state["item"])
            author_values = list(state["author"])
            length = min(len(item_values), max_length)
            if length:
                item_history[row_index, -length:] = item_values[-length:]
                author_history[row_index, -length:] = author_values[-length:]
                history_mask[row_index, -length:] = True
            if split_name == "train":
                state["item"].append(int(categorical_ids[row_index, 1]))
                state["author"].append(int(categorical_ids[row_index, 2]))
        payload["inputs"].update(
            {
                "history_item_ids": item_history,
                "history_author_ids": author_history,
                "history_mask": history_mask,
            }
        )
    return schema


def _exposure_key_from_raw(row):
    return (int(row[0]), str(row[1]), str(row[2]), str(row[4]), float(row[5]))


def _exposure_key_from_csv(row):
    return (
        int(row["date"]),
        str(row["user_id"]),
        str(row["video_id"]),
        str(row["tab"]),
        float(row["duration_ms"]),
    )


def load_train_auxiliary_targets(
    data_dir: str | Path,
    train_rows: Sequence,
    fields: Iterable[str],
):
    """Align real auxiliary labels from the training-window log only."""
    requested = tuple(dict.fromkeys(map(str, fields)))
    unsupported = set(requested) - TRAIN_AUXILIARY_FIELDS
    if unsupported:
        raise ValueError("unsupported auxiliary fields: " + ", ".join(sorted(unsupported)))
    source = Path(data_dir) / "log_standard_4_08_to_4_21_pure.csv"
    queues = defaultdict(deque)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(requested) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("training log lacks auxiliary fields: " + ", ".join(sorted(missing)))
        for row in reader:
            queues[_exposure_key_from_csv(row)].append(
                tuple(float(row[field] or 0.0) for field in requested)
            )
    values = {field: np.empty(len(train_rows), dtype=np.float32) for field in requested}
    for index, row in enumerate(train_rows):
        key = _exposure_key_from_raw(row)
        if not queues[key]:
            raise ValueError(f"cannot align train auxiliary row {index}: {key}")
        aligned = queues[key].popleft()
        for field, value in zip(requested, aligned):
            values[field][index] = value
    return values


def attach_train_auxiliary_targets(schema, targets: Mapping[str, np.ndarray]):
    """Attach auxiliary supervision to train only; validation/test stay clean."""
    train_payload = schema["splits"].get("train")
    if train_payload is None:
        raise ValueError("schema has no train split")
    expected = len(train_payload["targets"]["long_view"])
    for name, values in targets.items():
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (expected,):
            raise ValueError(f"auxiliary target {name!r} has shape {array.shape}, expected {(expected,)}")
        train_payload["targets"][str(name)] = array
    return schema
