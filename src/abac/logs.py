"""Access log generation from a ground-truth policy P*.

An access log L = {(q_i, l_i)} is a multiset of labelled requests (Section 2). For our
ground-truth ABAC datasets we can afford to enumerate the full request space U x O x R
and evaluate P* exactly, then treat the resulting grants as the "complete" log
(beta = 0, no missing observations) or subsample it to simulate a sparse/incomplete
log, which is the regime Rhapsody-style reliability miners are built for.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from .model import Policy

Request = Tuple[str, str, str]  # (uid, op, rid)


@dataclass
class AccessLog:
    """A concrete log: the full labelled request set, split into grants and denials."""

    grants: List[Request]
    denials: List[Request]

    def __len__(self) -> int:
        return len(self.grants) + len(self.denials)


def enumerate_full_log(policy: Policy) -> AccessLog:
    """Evaluate P* on the entire U x O x R space (exact ground truth, no sampling).

    Feasible for the ABAC-Lab-scale datasets used here (tens to hundreds of users and
    resources); for a larger deployment this would be replaced by a real observed log.
    """
    grants: List[Request] = []
    denials: List[Request] = []
    ops = sorted(policy.actions())
    uids = sorted(policy.users)
    rids = sorted(policy.resources)
    for uid in uids:
        for op in ops:
            for rid in rids:
                if policy.decide(uid, op, rid):
                    grants.append((uid, op, rid))
                else:
                    denials.append((uid, op, rid))
    return AccessLog(grants=grants, denials=denials)


def sample_sparse_log(full: AccessLog, keep_fraction: float, seed: int = 0) -> List[Request]:
    """Subsample the granted requests to simulate an incomplete observed log.

    Only grants are kept (a realistic access log records what happened -- permitted
    actions -- not the full space of what was denied or never attempted).

    Caution for attack experiments: uniform random subsampling can accidentally drop
    every entry referencing some entity, making that entity "unobserved" under a
    Rhapsody-style miner's closed-world universe -- which trivially satisfies
    reliability regardless of any threshold, for reasons having nothing to do with
    poisoning. `build_partial_log` avoids this confound.
    """
    rng = random.Random(seed)
    n_keep = max(1, int(round(len(full.grants) * keep_fraction)))
    return rng.sample(full.grants, k=min(n_keep, len(full.grants)))


def build_partial_log(
    full: AccessLog,
    focus_uids: "set[str]",
    focus_op: str,
    withhold_fraction: float,
    seed: int = 0,
) -> List[Request]:
    """The complete clean log, except a fraction of `focus_uids`' own `focus_op` grants
    are withheld -- i.e. permissions they hold but simply haven't exercised yet.
    Everything else (every other user, every other operation) stays fully observed.

    This isolates exactly the mechanism Section 1 describes -- an insider steers the log
    by choosing WHICH of their already-held permissions to exercise -- from incidental
    sparsity elsewhere in the log, which is a separate (and separately real) phenomenon
    but not what a legitimate-only poisoning attack needs to demonstrate.
    """
    rng = random.Random(seed)
    focus_grants = [q for q in full.grants if q[0] in focus_uids and q[1] == focus_op]
    n_withhold = int(round(withhold_fraction * len(focus_grants)))
    withhold = set(rng.sample(focus_grants, k=min(n_withhold, len(focus_grants))))
    return [q for q in full.grants if q not in withhold]
