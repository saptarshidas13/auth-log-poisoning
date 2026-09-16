"""Rhapsody-style sparse-log ABAC miner: tolerates incompleteness via a reliability threshold.

Same bottom-up engine as xu_stoller.py, but (a) the universe used to judge a
generalization is restricted to entities actually seen in the log (a sparse log
legitimately never mentions most of the population, so it would be unfair -- and would
force absurdly specific rules -- to demand soundness against never-observed entities),
and (b) a rule is accepted once its reliability clears `min_reliability` (default 0.9)
rather than requiring perfect (1.0) soundness. Lowering `min_reliability` makes the
miner generalize more aggressively over a sparse log -- exactly the dial Section 4.1
identifies as the mechanism a legitimate-only adversary exploits.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

from abac.model import Entity, Policy
from .common import mine_policy

Request = Tuple[str, str, str]


def mine(
    users: Dict[str, Entity],
    resources: Dict[str, Entity],
    log_grants: Sequence[Request],
    min_reliability: float = 0.9,
) -> Policy:
    return mine_policy(users, resources, log_grants, min_reliability=min_reliability, universe="observed")
