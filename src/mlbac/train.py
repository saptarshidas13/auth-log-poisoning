"""Training and evaluation for MLBACClassifier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .data import DLBACDataset
from .model import MLBACClassifier


def _to_tensors(ds: DLBACDataset):
    return (
        torch.from_numpy(ds.user_attrs),
        torch.from_numpy(ds.res_attrs),
        torch.from_numpy(ds.labels),
    )


def train_mlbac(
    train: DLBACDataset,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    hash_buckets: int = 4096,
    embed_dim: int = 16,
    hidden: int = 128,
    seed: int = 0,
) -> MLBACClassifier:
    torch.manual_seed(seed)
    model = MLBACClassifier(train.num_user_attrs, train.num_res_attrs, train.num_ops, hash_buckets, embed_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ua, ra, y = _to_tensors(train)
    n = len(train)
    rng = np.random.RandomState(seed)

    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_idx = torch.from_numpy(idx)
            logits = model(ua[batch_idx], ra[batch_idx])
            loss = loss_fn(logits, y[batch_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@dataclass
class EvalReport:
    accuracy_per_op: np.ndarray  # (num_ops,)
    overall_accuracy: float


def evaluate(model: MLBACClassifier, ds: DLBACDataset) -> EvalReport:
    ua, ra, y = _to_tensors(ds)
    with torch.no_grad():
        logits = model(ua, ra)
        pred = (torch.sigmoid(logits) >= 0.5).float()
    correct = (pred == y).numpy()
    acc_per_op = correct.mean(axis=0)
    return EvalReport(accuracy_per_op=acc_per_op, overall_accuracy=float(correct.mean()))


def predict_one(model: MLBACClassifier, user_attrs, res_attrs) -> np.ndarray:
    """Predict grant/deny for a single (user_attrs, res_attrs) pair. Returns shape (num_ops,) of 0/1."""
    ua = torch.tensor([user_attrs], dtype=torch.long)
    ra = torch.tensor([res_attrs], dtype=torch.long)
    with torch.no_grad():
        logits = model(ua, ra)
        pred = (torch.sigmoid(logits) >= 0.5).float()
    return pred.numpy()[0]
