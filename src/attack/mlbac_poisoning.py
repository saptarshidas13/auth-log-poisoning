"""Problem 2 (Section 4.2, C3): decision-boundary poisoning of MLBAC/DLBAC.

Black-box variant: "A shifts the learned decision boundary using only correctly
labelled grant examples clustered near t* in metadata space (a clean-label poisoning
setting specialized to authorization). ... black-box A uses metadata proximity to t*."

These DLBAC datasets are fixed samples, not a queryable ground-truth policy, so
"legitimate-only" here means every poisoned row is an EXISTING true-grant (label=1) row
already belonging to a controlled uid -- the adversary cannot fabricate a new attribute
combination, only choose how many times to replay (oversample) actions they already
legitimately performed. Oversampling near-target rows is a standard clean-label
poisoning technique: it doesn't change any label, only the empirical density the
classifier is trained against, which is exactly the resource this attack exploits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from mlbac.data import DLBACDataset, concat
from mlbac.model import MLBACClassifier
from mlbac.train import predict_one, train_mlbac
from .threat_model import Adversary


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def legitimate_action_rows(train: DLBACDataset, controlled_uids: Set[int], op_index: int) -> np.ndarray:
    """Indices of rows already belonging to a controlled user with a TRUE grant for op_index."""
    mask = np.isin(train.uids, list(controlled_uids)) & (train.labels[:, op_index] == 1.0)
    return np.nonzero(mask)[0]


class IllegitimateMLBACPoisonRecord(ValueError):
    pass


def validate_mlbac_poison(indices: Sequence[int], train: DLBACDataset, adversary: Adversary, op_index: int) -> None:
    controlled = {int(u) for u in adversary.controlled_uids}
    for i in indices:
        if int(train.uids[i]) not in controlled:
            raise IllegitimateMLBACPoisonRecord(f"row {i}: uid {train.uids[i]} not controlled by adversary")
        if train.labels[i, op_index] != 1.0:
            raise IllegitimateMLBACPoisonRecord(f"row {i}: not a true grant for op {op_index}")


@dataclass
class MLBACAttackTrace:
    poisoned_train: DLBACDataset
    poison_size: int
    baseline_prediction: int  # 0 or 1, target's predicted label with NO poisoning
    poisoned_prediction: int
    success: bool  # poisoned_prediction == 1 and baseline_prediction == 0


def _oversample_and_evaluate(
    train: DLBACDataset,
    chosen: Sequence[int],
    oversample_factor: int,
    max_size: int,
    target_user_attrs: Sequence[int],
    target_res_attrs: Sequence[int],
    op_index: int,
    adversary: Adversary,
    train_kwargs: dict,
    seed: int,
    baseline_pred: int,
) -> MLBACAttackTrace:
    validate_mlbac_poison(chosen, train, adversary, op_index)

    extra_rows: List[Tuple[int, int, List[int], List[int], List[float]]] = []
    for i in chosen:
        for _ in range(oversample_factor):
            if len(extra_rows) >= max_size:
                break
            extra_rows.append((
                int(train.uids[i]), int(train.rids[i]),
                train.user_attrs[i].tolist(), train.res_attrs[i].tolist(),
                train.labels[i].tolist(),
            ))

    poisoned_train = concat(train, extra_rows)
    poisoned_model = train_mlbac(poisoned_train, seed=seed, **train_kwargs)
    poisoned_pred = predict_one(poisoned_model, list(target_user_attrs), list(target_res_attrs))[op_index]

    return MLBACAttackTrace(
        poisoned_train=poisoned_train,
        poison_size=len(extra_rows),
        baseline_prediction=int(baseline_pred),
        poisoned_prediction=int(poisoned_pred),
        success=(poisoned_pred == 1 and baseline_pred == 0),
    )


def black_box_mlbac_attack(
    train: DLBACDataset,
    target_user_attrs: Sequence[int],
    target_res_attrs: Sequence[int],
    op_index: int,
    adversary: Adversary,
    k_neighbors: int,
    oversample_factor: int,
    train_kwargs: dict = None,
    seed: int = 0,
) -> MLBACAttackTrace:
    """Rank the adversary's own true-grant rows for op_index by resource-attribute
    proximity to the target, oversample the k_neighbors closest by `oversample_factor`,
    retrain, and check whether the target flipped -- against a same-seed baseline
    (no poison) so a flip is attributable to Delta, not training variance.
    """
    train_kwargs = train_kwargs or {}
    controlled = {int(u) for u in adversary.controlled_uids}

    baseline_model = train_mlbac(train, seed=seed, **train_kwargs)
    baseline_pred = predict_one(baseline_model, list(target_user_attrs), list(target_res_attrs))[op_index]

    candidate_idx = legitimate_action_rows(train, controlled, op_index)
    target_res = np.asarray(target_res_attrs)
    distances = [hamming_distance(train.res_attrs[i], target_res) for i in candidate_idx]
    order = np.argsort(distances)
    ranked = candidate_idx[order]
    chosen = ranked[:k_neighbors]

    max_size = int(adversary.beta * len(train))
    return _oversample_and_evaluate(
        train, chosen, oversample_factor, max_size, target_user_attrs, target_res_attrs,
        op_index, adversary, train_kwargs, seed, int(baseline_pred),
    )


def gradient_alignment_scores(
    model: MLBACClassifier,
    train: DLBACDataset,
    candidate_idx: np.ndarray,
    target_user_attrs: Sequence[int],
    target_res_attrs: Sequence[int],
    op_index: int,
) -> np.ndarray:
    """White-box ranking signal (Section 4.2: "white-box A uses influence or gradient
    signals"). A first-order, TracIn-style influence approximation with no Hessian:
    for each candidate, the cosine similarity between (a) the gradient of ITS OWN
    training loss (pushing the model to fit its true label=1), and (b) the gradient
    that would reduce the model's loss if the TARGET were labelled grant=1. A high
    score means training more on that candidate moves parameters in a direction that
    ALSO reduces loss on granting the target -- exactly the examples worth oversampling.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]

    def grad_for(user_attrs, res_attrs, label_vec) -> torch.Tensor:
        model.zero_grad()
        ua = torch.tensor([user_attrs], dtype=torch.long)
        ra = torch.tensor([res_attrs], dtype=torch.long)
        y = torch.tensor([label_vec], dtype=torch.float32)
        loss = loss_fn(model(ua, ra), y)
        grads = torch.autograd.grad(loss, params)
        return torch.cat([g.reshape(-1) for g in grads])

    target_label = [0.0] * train.num_ops
    target_label[op_index] = 1.0
    g_target = grad_for(list(target_user_attrs), list(target_res_attrs), target_label)
    g_target_norm = g_target.norm() + 1e-12

    scores = np.zeros(len(candidate_idx))
    for j, i in enumerate(candidate_idx):
        g_cand = grad_for(train.user_attrs[i].tolist(), train.res_attrs[i].tolist(), train.labels[i].tolist())
        cos = torch.dot(g_target, g_cand) / (g_target_norm * (g_cand.norm() + 1e-12))
        scores[j] = cos.item()
    return scores


def white_box_mlbac_attack(
    train: DLBACDataset,
    target_user_attrs: Sequence[int],
    target_res_attrs: Sequence[int],
    op_index: int,
    adversary: Adversary,
    k_neighbors: int,
    oversample_factor: int,
    train_kwargs: dict = None,
    seed: int = 0,
) -> MLBACAttackTrace:
    """Same clean-label oversampling mechanism as the black-box attack, but candidates
    are ranked by gradient alignment with the target (`gradient_alignment_scores`)
    instead of raw metadata distance -- the white-box variant Section 4.2 names.
    """
    train_kwargs = train_kwargs or {}
    controlled = {int(u) for u in adversary.controlled_uids}

    baseline_model = train_mlbac(train, seed=seed, **train_kwargs)
    baseline_pred = predict_one(baseline_model, list(target_user_attrs), list(target_res_attrs))[op_index]

    candidate_idx = legitimate_action_rows(train, controlled, op_index)
    scores = gradient_alignment_scores(baseline_model, train, candidate_idx, target_user_attrs, target_res_attrs, op_index)
    order = np.argsort(-scores)  # highest alignment first
    ranked = candidate_idx[order]
    chosen = ranked[:k_neighbors]

    max_size = int(adversary.beta * len(train))
    return _oversample_and_evaluate(
        train, chosen, oversample_factor, max_size, target_user_attrs, target_res_attrs,
        op_index, adversary, train_kwargs, seed, int(baseline_pred),
    )
