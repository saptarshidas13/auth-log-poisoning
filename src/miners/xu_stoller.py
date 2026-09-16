"""Xu-Stoller-style strict bottom-up ABAC miner: requires perfectly sound rules.

This is the Mine that Problem 1 attacks in its "no tolerance for sparsity" form: with
min_reliability=1.0 and universe="full", a generalization is accepted only if it grants
nothing beyond what the (assumed complete) log already shows.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

from abac.model import Entity, Policy
from .common import mine_policy

Request = Tuple[str, str, str]


def mine(users: Dict[str, Entity], resources: Dict[str, Entity], log_grants: Sequence[Request]) -> Policy:
    return mine_policy(users, resources, log_grants, min_reliability=1.0, universe="full")
