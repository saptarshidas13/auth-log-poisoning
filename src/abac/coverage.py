"""Efficient evaluation of a Policy's full grant set (indexed, not brute-force).

Used for over-privilege/under-privilege comparison against ground truth: instead of
looping over every (u, op, r) request and testing every rule against it (O(requests x
rules), which blows up once a mined policy has thousands of rules), compute each rule's
coverage via attribute indices and union the results (O(rules x coverage)).
"""
from __future__ import annotations

from typing import FrozenSet, Optional, Set, Tuple

from .index import AttrIndex
from .model import Policy
from .schema import single_valued_attrs

Request = Tuple[str, str, str]


def policy_grants(
    policy: Policy,
    universe_u_ids: Optional[FrozenSet[str]] = None,
    universe_r_ids: Optional[FrozenSet[str]] = None,
) -> Set[Request]:
    user_attrs = single_valued_attrs(policy.users, exclude=set())
    res_attrs = single_valued_attrs(policy.resources, exclude=set())
    u_index = AttrIndex(policy.users, user_attrs)
    r_index = AttrIndex(policy.resources, res_attrs)
    uu = universe_u_ids if universe_u_ids is not None else u_index.all_ids
    ur = universe_r_ids if universe_r_ids is not None else r_index.all_ids

    grants: Set[Request] = set()
    for rule in policy.rules:
        mu = u_index.matched_ids(rule.sub_cond) & uu
        mr = r_index.matched_ids(rule.res_cond) & ur
        if not mu or not mr:
            continue
        if not rule.cons:
            pairs = [(u, r) for u in mu for r in mr]
        else:
            pairs = []
            mu_entities = [(uid, policy.users[uid]) for uid in mu]
            mr_entities = [(rid, policy.resources[rid]) for rid in mr]
            for uid, u in mu_entities:
                for rid, r in mr_entities:
                    if rule.matches_constraints(u, r):
                        pairs.append((uid, rid))
        for op in rule.acts:
            for uid, rid in pairs:
                grants.add((uid, op, rid))
    return grants
