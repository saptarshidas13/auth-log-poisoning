"""Restricter-style log-driven least-privilege tightening (Section 4.3, Problem 3 / C4).

Unlike the rule miners, real tightening tools (AWS IAM Access Analyzer,
IAM-PolicyRefiner, Restricter) don't synthesize attribute rules -- they work over an
existing, already-deployed policy P0 and trim permissions that observed activity never
exercised. We model P0 and the tightened output as explicit permission sets, matching
how these tools actually operate (they diff a policy's statements against CloudTrail-
style usage logs, they don't re-derive ABAC conditions).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Sequence, Set, Tuple

from abac.coverage import policy_grants
from abac.model import Policy

Request = Tuple[str, str, str]


@dataclass
class TightenedPolicy:
    grants: Set[Request]

    def decide(self, uid: str, op: str, rid: str) -> int:
        return 1 if (uid, op, rid) in self.grants else 0


def mine(p0_grants: Set[Request], log: Sequence[Request], min_uses: int = 1) -> TightenedPolicy:
    """Keep a P0 permission only if it was exercised at least `min_uses` times in `log`.

    min_uses=1 is the simplest "ever used, keep it" rule; a higher threshold models
    tools that require a permission to be exercised repeatedly (not just once, possibly
    by accident) before trusting it as genuinely needed.
    """
    usage = Counter(log)
    retained = {q for q in p0_grants if usage[q] >= min_uses}
    return TightenedPolicy(grants=retained)


def build_role_based_overprivileged_policy(policy_star: Policy, role_attr: str = "position") -> Set[Request]:
    """A realistic over-privileged starting policy P0, always a superset of P*'s grants.

    Models the common real-world provisioning pattern of copying an existing "template"
    user's access when onboarding a new account, rather than deriving access
    attribute-by-attribute: whatever ANY user sharing `role_attr`'s value is truly
    granted under P*, every user with that same value is ALSO granted under P0. The
    resulting excess (P0 minus P*'s own grants) is exactly the kind of over-privilege a
    least-privilege tightening tool is meant to remove.
    """
    true_grants = policy_grants(policy_star)

    role_of: Dict[str, str] = {uid: e.attrs.get(role_attr) for uid, e in policy_star.users.items()}
    by_role_grants: Dict[str, Set[Tuple[str, str]]] = {}
    for uid, op, rid in true_grants:
        role = role_of.get(uid)
        if role is None:
            continue
        by_role_grants.setdefault(role, set()).add((op, rid))

    p0: Set[Request] = set()
    for uid, role in role_of.items():
        for op, rid in by_role_grants.get(role, ()):
            p0.add((uid, op, rid))
    return p0
