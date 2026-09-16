"""Loader for the DLBAC_alpha dataset format (Nobi et al., CODASPY 2022).

Each line: uid rid <user_metadata...> <resource_metadata...> <op1 op2 ... opK>
Synthetic datasets: symmetric user/resource metadata counts, K=4 operations.
Real-world (amazon1/2/3): 8 user-metadata columns, 1 resource-metadata column, K=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# name -> (num_user_attrs, num_res_attrs, num_ops), inferred from each file's column
# count (see datasets/.../ReadMe.md): total_cols = 2 (uid,rid) + user + res + ops.
SYNTHETIC_DATASETS = {
    "u4k-r4k-auth11k": (8, 8, 4),
    "u5k-r5k-auth12k": (8, 8, 4),
    "u5k-r5k-auth19k": (10, 10, 4),
    "u6k-r6k-auth32k": (10, 10, 4),
    "u4k-r4k-auth21k": (11, 11, 4),
    "u4k-r7k-auth20k": (11, 11, 4),
    "u4k-r4k-auth22k": (13, 13, 4),
    "u4k-r6k-auth28k": (13, 13, 4),
}
REAL_WORLD_DATASETS = {
    "amazon1": (8, 1, 1),
    # amazon2/amazon3 are augmented instances (Nobi et al. 2022, Sec 4.1) with more
    # resource metadata than amazon1 -- 16 columns/row (2 ids + 8 user-attrs +
    # 5 res-attrs + 1 op), confirmed by inspecting the actual sample files
    # (DlbacAlpha's ReadMe doesn't state the exact split).
    "amazon2": (8, 5, 1),
    "amazon3": (8, 5, 1),
}


@dataclass
class DLBACDataset:
    uids: np.ndarray  # (N,) int
    rids: np.ndarray  # (N,) int
    user_attrs: np.ndarray  # (N, num_user_attrs) int
    res_attrs: np.ndarray  # (N, num_res_attrs) int
    labels: np.ndarray  # (N, num_ops) float32, 0/1
    num_user_attrs: int
    num_res_attrs: int
    num_ops: int

    def __len__(self) -> int:
        return len(self.uids)


def load_sample_file(path: str, num_user_attrs: int, num_res_attrs: int, num_ops: int) -> DLBACDataset:
    uids, rids, user_attrs, res_attrs, labels = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            expected = 2 + num_user_attrs + num_res_attrs + num_ops
            if len(parts) != expected:
                raise ValueError(f"{path}: expected {expected} columns, got {len(parts)}: {line!r}")
            vals = [int(x) for x in parts]
            uids.append(vals[0])
            rids.append(vals[1])
            i = 2
            user_attrs.append(vals[i:i + num_user_attrs]); i += num_user_attrs
            res_attrs.append(vals[i:i + num_res_attrs]); i += num_res_attrs
            labels.append(vals[i:i + num_ops])
    return DLBACDataset(
        uids=np.array(uids, dtype=np.int64),
        rids=np.array(rids, dtype=np.int64),
        user_attrs=np.array(user_attrs, dtype=np.int64),
        res_attrs=np.array(res_attrs, dtype=np.int64),
        labels=np.array(labels, dtype=np.float32),
        num_user_attrs=num_user_attrs,
        num_res_attrs=num_res_attrs,
        num_ops=num_ops,
    )


def load_dlbac(name: str, dataset_root: str) -> Tuple[DLBACDataset, DLBACDataset]:
    """Load (train, test) for a named synthetic or real-world DLBAC dataset."""
    import os

    if name in SYNTHETIC_DATASETS:
        cfg = SYNTHETIC_DATASETS[name]
        base = os.path.join(dataset_root, "synthetic", name)
    elif name in REAL_WORLD_DATASETS:
        cfg = REAL_WORLD_DATASETS[name]
        base = os.path.join(dataset_root, "real-world", name)
    else:
        raise KeyError(f"unknown DLBAC dataset {name!r}")

    num_user_attrs, num_res_attrs, num_ops = cfg
    train = load_sample_file(os.path.join(base, f"train_{name}.sample"), num_user_attrs, num_res_attrs, num_ops)
    test = load_sample_file(os.path.join(base, f"test_{name}.sample"), num_user_attrs, num_res_attrs, num_ops)
    return train, test


def concat(a: DLBACDataset, extra_rows: List[Tuple[int, int, List[int], List[int], List[float]]]) -> DLBACDataset:
    """Append extra (uid, rid, user_attrs, res_attrs, labels) rows to a dataset (for poisoning)."""
    if not extra_rows:
        return a
    uids = np.concatenate([a.uids, np.array([r[0] for r in extra_rows], dtype=np.int64)])
    rids = np.concatenate([a.rids, np.array([r[1] for r in extra_rows], dtype=np.int64)])
    user_attrs = np.concatenate([a.user_attrs, np.array([r[2] for r in extra_rows], dtype=np.int64)])
    res_attrs = np.concatenate([a.res_attrs, np.array([r[3] for r in extra_rows], dtype=np.int64)])
    labels = np.concatenate([a.labels, np.array([r[4] for r in extra_rows], dtype=np.float32)])
    return DLBACDataset(uids, rids, user_attrs, res_attrs, labels, a.num_user_attrs, a.num_res_attrs, a.num_ops)
