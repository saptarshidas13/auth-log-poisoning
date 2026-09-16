"""Before/after attack evaluation, operationalizing the metrics of Section 6.3:

    attack success rate on t*; over-privilege gained per poison record; poison budget
    beta needed for success; collateral decision change on a clean validation log;
    stealth metrics (rule-count delta, conjunct-count delta as a reliability/complexity
    proxy, validation-accuracy delta).

A Phase 3 attack only needs to produce a `Delta` (validated by threat_model) and call
`run_attack_trial`; every metric the evaluation plan asks for falls out of one before/
after comparison against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

from abac.logs import AccessLog
from abac.model import Entity, Policy
from metrics import compare_to_ground_truth, policy_size
from .threat_model import Adversary, apply_poison, validate_poison_set

Request = Tuple[str, str, str]
MineFn = Callable[[Dict[str, Entity], Dict[str, Entity], Sequence[Request]], Policy]


@dataclass
class AttackResult:
    success: bool
    target: Request

    over_privilege_before: int
    over_privilege_after: int
    under_privilege_before: int
    under_privilege_after: int
    collateral_over_privilege: int  # over-privilege_after, excluding the target itself

    rules_before: int
    rules_after: int
    conjuncts_before: int
    conjuncts_after: int

    poison_size: int
    poison_fraction: float  # |Delta| / |L|

    @property
    def over_privilege_gained(self) -> int:
        return self.over_privilege_after - self.over_privilege_before

    @property
    def over_privilege_gained_per_poison_record(self) -> float:
        return self.over_privilege_gained / self.poison_size if self.poison_size else 0.0

    @property
    def rule_count_delta(self) -> int:
        return self.rules_after - self.rules_before

    @property
    def conjunct_count_delta(self) -> int:
        return self.conjuncts_after - self.conjuncts_before


def run_attack_trial(
    mine_fn: MineFn,
    users: Dict[str, Entity],
    resources: Dict[str, Entity],
    base_log: Sequence[Request],
    delta: Sequence[Request],
    target: Request,
    ground_truth_full_log: AccessLog,
    policy_star=None,
    adversary: Adversary = None,
) -> AttackResult:
    """Mine on `base_log` and on `base_log u delta`, compare both against ground truth,
    and report whether `target` flipped from denied to granted.

    If `policy_star` and `adversary` are given, `delta` is validated against the
    legitimate-only + budget restriction first (recommended for every real attack run;
    optional only for quick machinery smoke tests that construct a deliberately invalid
    Delta to confirm the check fires).
    """
    if policy_star is not None and adversary is not None:
        validate_poison_set(delta, policy_star, adversary, base_log_size=len(base_log))

    policy_before = mine_fn(users, resources, base_log)
    poisoned_log = apply_poison(base_log, delta)
    policy_after = mine_fn(users, resources, poisoned_log)

    report_before = compare_to_ground_truth(policy_before, ground_truth_full_log)
    report_after = compare_to_ground_truth(policy_after, ground_truth_full_log)

    uid, op, rid = target
    decided_before = policy_before.decide(uid, op, rid)
    decided_after = policy_after.decide(uid, op, rid)
    success = decided_before == 0 and decided_after == 1

    collateral = report_after.over_privilege - {target}

    rules_before, conjuncts_before = policy_size(policy_before)
    rules_after, conjuncts_after = policy_size(policy_after)

    return AttackResult(
        success=success,
        target=target,
        over_privilege_before=report_before.over_privilege_count,
        over_privilege_after=report_after.over_privilege_count,
        under_privilege_before=report_before.under_privilege_count,
        under_privilege_after=report_after.under_privilege_count,
        collateral_over_privilege=len(collateral),
        rules_before=rules_before,
        rules_after=rules_after,
        conjuncts_before=conjuncts_before,
        conjuncts_after=conjuncts_after,
        poison_size=len(delta),
        poison_fraction=len(delta) / len(base_log) if base_log else 0.0,
    )
