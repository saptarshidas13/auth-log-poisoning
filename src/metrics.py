"""Metrics for comparing a mined policy P_hat against ground truth P* (Section 2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple

from abac.coverage import policy_grants
from abac.logs import AccessLog
from abac.model import Policy

Request = Tuple[str, str, str]


@dataclass
class ComparisonReport:
    over_privilege: Set[Request]  # granted by P_hat, denied by P*
    under_privilege: Set[Request]  # denied by P_hat, granted by P*
    n_requests: int
    n_true_grants: int

    @property
    def over_privilege_count(self) -> int:
        return len(self.over_privilege)

    @property
    def under_privilege_count(self) -> int:
        return len(self.under_privilege)

    @property
    def exact_match(self) -> bool:
        return not self.over_privilege and not self.under_privilege


def compare_to_ground_truth(mined: Policy, ground_truth_full_log: AccessLog, ops=None) -> ComparisonReport:
    """Evaluate `mined` over the same full U x O x R space used to build `ground_truth_full_log`.

    Uses the indexed `policy_grants` evaluator (O(rules x coverage)) rather than looping
    over every request and testing every rule (O(requests x rules)) -- the latter is
    fine for a ground-truth policy with a few dozen rules but blows up for a mined
    policy that can have thousands (e.g. a sparse-log Rhapsody run).
    """
    true_grants = set(ground_truth_full_log.grants)
    true_denials = set(ground_truth_full_log.denials)

    mined_grants = policy_grants(mined)
    over = mined_grants & true_denials
    under = true_grants - mined_grants

    return ComparisonReport(
        over_privilege=over,
        under_privilege=under,
        n_requests=len(true_grants) + len(true_denials),
        n_true_grants=len(true_grants),
    )


def policy_size(policy: Policy) -> Tuple[int, int]:
    """(rule count, total conjunct count) -- the quality/compactness terms miners optimize."""
    return policy.num_rules(), policy.total_conjuncts()
