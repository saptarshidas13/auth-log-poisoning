"""Threat-model machinery for legitimate-only poisoning (Section 3 of the problem statement).

This module makes the adversary's defining restriction a hard invariant that any attack
built on top of it (Phase 3) cannot violate by construction:

    (q, l) in Delta  =>  l = auth_{P*}(q) = 1  and  q is executable by a subject A controls

i.e. every poison record is a genuine, correctly-labelled, permitted action by a user the
adversary already controls -- no forged records, no flipped labels, no denials inserted.
`validate_poison_set` checks this explicitly against the ground-truth policy P*, and
`apply_poison` is the only way to actually merge Delta into a log, so a Phase 3 attack
that only ever calls these two functions cannot accidentally cheat the threat model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Sequence, Tuple

from abac.coverage import policy_grants
from abac.model import Policy

Request = Tuple[str, str, str]

KNOWLEDGE_LEVELS = ("black_box", "grey_box", "white_box")


class IllegitimatePoisonRecord(ValueError):
    """Raised when a candidate poison record violates the legitimate-only restriction."""


class PoisonBudgetExceeded(ValueError):
    """Raised when |Delta| > beta * |L| (Section 3.1)."""


@dataclass(frozen=True)
class Adversary:
    """A insider or small colluding group (Section 3.1-3.2).

    controlled_uids: the subjects A can issue requests as.
    knowledge: one of "black_box" (own attributes + observed decisions only),
        "grey_box" (+ miner family/hyperparameters), "white_box" (+ the current log).
        This module doesn't gate information itself -- it documents which knowledge
        level a Phase 3 attack is claiming to operate under, since that claim is a
        property of what the ATTACK CODE looks at, not of the threat-model plumbing.
    beta: poison budget as a fraction of the base log size |L|.
    """

    controlled_uids: FrozenSet[str]
    knowledge: str = "black_box"
    beta: float = 0.05

    def __post_init__(self):
        if self.knowledge not in KNOWLEDGE_LEVELS:
            raise ValueError(f"knowledge must be one of {KNOWLEDGE_LEVELS}, got {self.knowledge!r}")
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")


def legitimate_action_space(policy_star: Policy, adversary: Adversary) -> List[Request]:
    """Every (uid, op, rid) the adversary's controlled users are already granted under P*.

    This is the entire pool an attack may draw poison records from -- by construction it
    contains nothing the adversary isn't already allowed to do.
    """
    controlled = frozenset(adversary.controlled_uids)
    grants = policy_grants(policy_star, universe_u_ids=controlled)
    return sorted(grants)


def validate_poison_set(
    delta: Sequence[Request],
    policy_star: Policy,
    adversary: Adversary,
    base_log_size: int,
) -> None:
    """Raise if `delta` violates the legitimate-only or budget restriction. No return value
    on success -- call this before trusting any Delta a Phase 3 attack produces."""
    max_size = adversary.beta * base_log_size
    if len(delta) > max_size:
        raise PoisonBudgetExceeded(
            f"|Delta|={len(delta)} exceeds beta*|L|={max_size:.1f} (beta={adversary.beta}, |L|={base_log_size})"
        )
    for uid, op, rid in delta:
        if uid not in adversary.controlled_uids:
            raise IllegitimatePoisonRecord(f"({uid}, {op}, {rid}): uid not controlled by adversary")
        if policy_star.decide(uid, op, rid) != 1:
            raise IllegitimatePoisonRecord(f"({uid}, {op}, {rid}): not a true grant under P*")


def apply_poison(base_log: Sequence[Request], delta: Sequence[Request]) -> List[Request]:
    """L u Delta, as a multiset (plain concatenation -- duplicates are meaningful)."""
    return list(base_log) + list(delta)


def is_valid_target(target: Request, policy_star: Policy) -> bool:
    """A well-formed attack target t* must be denied under P* (Section 3.3)."""
    uid, op, rid = target
    return policy_star.decide(uid, op, rid) == 0
