"""Trusted KuaiRand loader with a validation-only cache mode.

The original KuaiRand-Pure CSV path remains available for compatibility.  A
bonus-dataset run sets ``KUAIRAND_CACHE_PATH`` and receives only deterministic
train/validation views from a prebuilt NPZ archive; test labels are never
returned to candidate code.
"""
from __future__ import annotations

import collections
import csv
import os
from pathlib import Path

import numpy as np


LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
LEGACY_FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
CACHE_FIELDS = ["user_id", "video_id"]
FIELDS = CACHE_FIELDS if os.getenv("KUAIRAND_CACHE_PATH") else LEGACY_FIELDS
DATASET_NAME = os.getenv("KUAIRAND_DATASET_NAME", "KuaiRand-Pure")


class CachedSplit(collections.namedtuple(
    "CachedSplitBase",
    "users items labels history_end num_users num_items",
)):
    __slots__ = ()


def _sample_indices(length: int, limit: int, seed: int):
    if limit <= 0 or limit >= length:
        return slice(None)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=limit, replace=False))


def _load_cache(path: str | os.PathLike[str]):
    cache_path = Path(path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"{DATASET_NAME} cache not found: {cache_path}")
    seed = int(os.getenv("KUAIRAND_SAMPLE_SEED", "20260901"))
    limits = {
        "train": int(os.getenv("KUAIRAND_MAX_TRAIN_ROWS", "0")),
        "valid": int(os.getenv("KUAIRAND_MAX_VALID_ROWS", "0")),
    }
    result = {}
    with np.load(cache_path, allow_pickle=False) as archive:
        num_users = int(np.asarray(archive["num_users"]).item())
        num_items = int(np.asarray(archive["num_items"]).item())
        for offset, name in enumerate(("train", "valid")):
            labels = archive[f"{name}_labels"]
            selected = _sample_indices(len(labels), limits[name], seed + offset)
            result[name] = CachedSplit(
                users=np.asarray(archive[f"{name}_users"][selected], dtype=np.int32),
                items=np.asarray(archive[f"{name}_items"][selected], dtype=np.int32),
                labels=np.asarray(labels[selected], dtype=np.float32),
                history_end=np.asarray(
                    archive[f"{name}_history_end"][selected], dtype=np.int32
                ),
                num_users=num_users,
                num_items=num_items,
            )
    return result


def _load_legacy(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as handle:
        for row in csv.DictReader(handle):
            vid2author[row["video_id"]] = row["author_id"]
    rows = []
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, filename)) as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    (
                        int(row["date"]),
                        row["user_id"],
                        row["video_id"],
                        vid2author.get(row["video_id"], "UNK"),
                        row["tab"],
                        float(row["duration_ms"]),
                        1 if row[LABEL] != "0" else 0,
                    )
                )
    return {
        name: [row for row in rows if lo <= row[0] <= hi]
        for name, (lo, hi) in SPLITS.items()
    }


def load(data_dir):
    """Load cache train/valid when configured, otherwise the original CSVs."""
    cache_path = os.getenv("KUAIRAND_CACHE_PATH")
    return _load_cache(cache_path) if cache_path else _load_legacy(data_dir)


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def _encode_cache(splits, feature_state=None):
    if feature_state is None:
        reference = next(iter(splits.values()))
        item_values = np.unique(np.asarray(splits["train"].items, dtype=np.int32))
        feature_state = {
            "schema_version": 1,
            "fields": list(CACHE_FIELDS),
            "num_users": int(reference.num_users),
            "num_items": int(reference.num_items),
            "item_offset": int(reference.num_users) + 1,
            # Fit the compact vocabulary on train only.  The original cache
            # uses sparse global item IDs (up to tens of millions), which
            # would otherwise create multi-gigabyte embeddings/checkpoints.
            "item_values": item_values.tolist(),
        }
    elif feature_state.get("fields") != CACHE_FIELDS:
        raise ValueError("feature_state fields do not match cache-backed fields")
    item_offset = int(feature_state["item_offset"])
    item_values = np.asarray(feature_state["item_values"], dtype=np.int32)
    unknown_item = len(item_values)
    feature_dimension = item_offset + unknown_item + 1
    encoded = {}
    for name, split in splits.items():
        x = np.empty((len(split.labels), 2), dtype=np.int32)
        x[:, 0] = split.users
        positions = np.searchsorted(item_values, split.items)
        matched = positions < len(item_values)
        matched_indices = np.flatnonzero(matched)
        matched[matched_indices] = (
            item_values[positions[matched_indices]] == split.items[matched_indices]
        )
        compact_items = np.full(len(split.items), unknown_item, dtype=np.int32)
        compact_items[matched] = positions[matched]
        x[:, 1] = compact_items + item_offset
        encoded[name] = (x, split.labels, split.users)
    return encoded, feature_dimension, feature_state


def _encode_legacy(splits, feature_state=None):
    if feature_state is None:
        train = splits["train"]
        edges = _bucket_edges([row[5] for row in train])
    else:
        if feature_state.get("fields") != LEGACY_FIELDS:
            raise ValueError("feature_state fields do not match legacy fields")
        edges = np.asarray(feature_state["edges"], dtype=np.float64)

    def raw(row):
        return [
            row[1], row[2], row[3], row[4],
            str(int(np.searchsorted(edges, row[5]))),
        ]

    if feature_state is None:
        vocabs = [dict() for _ in LEGACY_FIELDS]
        for row in train:
            for index, value in enumerate(raw(row)):
                if value not in vocabs[index]:
                    vocabs[index][value] = len(vocabs[index])
        unknown = [len(vocab) for vocab in vocabs]
        field_dims = [len(vocab) + 1 for vocab in vocabs]
        offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
        feature_state = {
            "schema_version": 1,
            "fields": list(LEGACY_FIELDS),
            "edges": edges.tolist(),
            "vocabs": vocabs,
            "unk": unknown,
            "field_dims": field_dims,
            "offsets": offsets.tolist(),
        }
    else:
        vocabs = feature_state["vocabs"]
        unknown = [int(value) for value in feature_state["unk"]]
        field_dims = [int(value) for value in feature_state["field_dims"]]
        offsets = np.asarray(feature_state["offsets"], dtype=np.int32)
    encoded = {}
    for name, rows in splits.items():
        x = np.empty((len(rows), len(LEGACY_FIELDS)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for row_index, row in enumerate(rows):
            for field_index, value in enumerate(raw(row)):
                x[row_index, field_index] = (
                    vocabs[field_index].get(value, unknown[field_index])
                    + offsets[field_index]
                )
            y[row_index] = row[6]
            users.append(row[1])
        encoded[name] = (x, y, users)
    return encoded, int(sum(field_dims)), feature_state


def encode(splits, feature_state=None, return_state=False):
    """Encode a cache or legacy split with frozen feature state."""
    cache_mode = bool(splits) and isinstance(next(iter(splits.values())), CachedSplit)
    encoded, dimension, state = (
        _encode_cache(splits, feature_state)
        if cache_mode
        else _encode_legacy(splits, feature_state)
    )
    result = (encoded, dimension)
    return result + (state,) if return_state else result
