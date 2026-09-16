"""Problem 3 (Section 4.3, C4): privilege-retention poisoning against a log-driven
tightening tool. Rather than adding privilege (Problems 1/2), the adversary PREVENTS
the removal of an existing excess privilege simply by using it -- indistinguishable
from ordinary work, making this "the most deployable variant" (Section 4.3).

Legitimacy standard here is deliberately different from threat_model.py's
validate_poison_set. That function checks legitimacy against ground truth P*, correct
for Problems 1/2 where the adversary tries to gain something P* denies. Here the
adversary tries to KEEP something P0 -- the current, already-deployed, and by
construction over-privileged policy -- grants but P* would deny. Exercising it is
legitimate precisely because P0, not P*, is what actually governs access right now; the
insider isn't bypassing anything, just using access they already have.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set, Tuple

from tightening.restricter import mine
from .threat_model import Adversary

Request = Tuple[str, str, str]


class IllegitimateRetentionPoison(ValueError):
    """Raised when a candidate retention-poison record isn't currently permitted under P0,
    or isn't executable by a subject the adversary controls."""


def validate_retention_poison(delta: Sequence[Request], p0_grants: Set[Request], adversary: Adversary) -> None:
    for uid, op, rid in delta:
        if uid not in adversary.controlled_uids:
            raise IllegitimateRetentionPoison(f"({uid}, {op}, {rid}): uid not controlled by adversary")
        if (uid, op, rid) not in p0_grants:
            raise IllegitimateRetentionPoison(f"({uid}, {op}, {rid}): not currently permitted under P0")


@dataclass
class RetentionAttackResult:
    target: Request
    min_uses_required: int
    poison_size: int
    retained_without_poison: bool
    retained_with_poison: bool

    @property
    def success(self) -> bool:
        return self.retained_with_poison and not self.retained_without_poison


def retention_attack(
    p0_grants: Set[Request],
    base_log: Sequence[Request],
    target: Request,
    adversary: Adversary,
    min_uses: int = 3,
) -> RetentionAttackResult:
    """Minimal Delta: repeat `target` exactly `min_uses` times -- the tightening tool's
    own retention threshold. This is provably minimal: `mine`'s Counter-based rule means
    fewer than `min_uses` repeats can never cross a >= min_uses retention count.
    """
    before = mine(p0_grants, base_log, min_uses=min_uses)
    retained_before = target in before.grants

    delta = [target] * min_uses
    validate_retention_poison(delta, p0_grants, adversary)

    poisoned_log = list(base_log) + delta
    after = mine(p0_grants, poisoned_log, min_uses=min_uses)
    retained_after = target in after.grants

    return RetentionAttackResult(
        target=target,
        min_uses_required=min_uses,
        poison_size=len(delta),
        retained_without_poison=retained_before,
        retained_with_poison=retained_after,
    )
