"""Shared bottom-up ABAC rule-mining engine.

This is our own reimplementation, in the spirit of Xu & Stoller's bottom-up ABAC
policy miner (SACMAT'13 / TDSC'15) and their sparse-log "Rhapsody" reliability miner
(DBSec'14) -- it is not a byte-for-byte port of their Java code. Both miner families
share one mechanism here, the thing Problem 1 (Section 4.1) exploits:

    1. For each (user, resource) pair granted some operation in the log, build the
       maximally SPECIFIC candidate rule that explains it (every attribute of the
       user and resource turned into a conjunct, plus every relational constraint
       between their attributes that happens to hold).
    2. GENERALIZE by greedily dropping conjuncts, as long as the rule stays
       "acceptable": for Xu-Stoller-style strict mining that means the rule must be
       perfectly SOUND (reliability == 1.0, i.e. it grants nothing outside the
       observed ACL, checked via closed-world assumption over a chosen universe of
       entities); for Rhapsody-style sparse-log mining, a rule is accepted once its
       RELIABILITY (fraction of everything it would grant that is actually attested
       in the log) clears a threshold < 1.0, which is exactly the knob a
       legitimate-only adversary can exploit to make the miner over-generalize
       (Section 4.1: "Rhapsody's reliability measure ... governs how aggressively"
       the miner generalizes).
    3. Repeat until every granted pair for that operation is covered by some rule.
    4. Compact: rules that differ only in their action set are merged.

Known v1 simplification (documented, not hidden): generalization only ever DROPS a
whole conjunct; it never widens an `InCond` value set by unioning two users' values
(e.g. turning `department [ {cs}` and `department [ {ee}` into one rule with
`department [ {cs ee}`). This still lets equivalent tuples merge into shared rules via
the covering loop, just less compactly than an optimal miner. Value-set widening is a
natural extension point, left for later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from abac.index import AttrIndex
from abac.model import (
    Entity,
    EqualityConstraint,
    InCond,
    Policy,
    ResourceInUserConstraint,
    Rule,
    SupersetConstraint,
    UserInResourceConstraint,
)
from abac.schema import multi_valued_attrs, single_valued_attrs

Request = Tuple[str, str, str]


@dataclass
class MinerConfig:
    min_reliability: float = 1.0  # 1.0 = strict Xu-Stoller-style soundness
    universe: str = "full"  # "full" = all U x R; "observed" = only entities seen in the log
    max_rules_per_op: int = 100_000  # safety valve against runaway loops on odd inputs


class BottomUpABACMiner:
    def __init__(self, users: Dict[str, Entity], resources: Dict[str, Entity], config: MinerConfig):
        self.users = users
        self.resources = resources
        self.config = config
        # InCond candidates for subCond/resCond exclude the identifier attribute --
        # `uid [ {csStu4}` would be a degenerate single-user rule.
        self.single_user_attrs = single_valued_attrs(users, exclude={"uid"})
        self.single_res_attrs = single_valued_attrs(resources, exclude={"rid"})
        self.multi_user_attrs = multi_valued_attrs(users, exclude=set())
        self.multi_res_attrs = multi_valued_attrs(resources, exclude=set())
        # Relational-constraint candidates DO include uid/rid: ownership patterns like
        # "a user can read their own transcript" (uid = student) or an ACL attribute
        # containing a specific uid are core, non-degenerate ABAC idioms that only make
        # sense as a *relation* between the two entities, never as a standalone
        # subCond/resCond conjunct.
        self.cons_user_attrs = single_valued_attrs(users, exclude=set())
        self.cons_res_attrs = single_valued_attrs(resources, exclude=set())

        # Attribute-value -> id-set indices, so matching a subCond/resCond against the
        # whole population is a handful of C-level set intersections instead of a
        # per-entity Python attribute lookup. This is what makes mining tractable on
        # the larger datasets (workforce: 353x250, edocument: 500x300).
        self._user_idx = AttrIndex(users, self.single_user_attrs)
        self._res_idx = AttrIndex(resources, self.single_res_attrs)
        self._all_user_ids = self._user_idx.all_ids
        self._all_res_ids = self._res_idx.all_ids

    # -- candidate rule construction -------------------------------------------------

    def _specific_rule(self, uid: str, rid: str, op: str) -> Rule:
        u, r = self.users[uid], self.resources[rid]
        sub_cond = [
            InCond(a, frozenset({u.attrs[a]})) for a in self.single_user_attrs if a in u.attrs
        ]
        res_cond = [
            InCond(a, frozenset({r.attrs[a]})) for a in self.single_res_attrs if a in r.attrs
        ]
        cons: List = []
        for ua in self.cons_user_attrs:
            uv = u.attrs.get(ua)
            if uv is None:
                continue
            for ra in self.cons_res_attrs:
                if r.attrs.get(ra) == uv:
                    cons.append(EqualityConstraint(ua, ra))
            for ra in self.multi_res_attrs:
                rv = r.attrs.get(ra)
                if isinstance(rv, frozenset) and uv in rv:
                    cons.append(UserInResourceConstraint(ua, ra))
        for ua in self.multi_user_attrs:
            uv = u.attrs.get(ua)
            if not isinstance(uv, frozenset) or not uv:
                continue
            for ra in self.cons_res_attrs:
                rv = r.attrs.get(ra)
                if rv is not None and rv in uv:
                    cons.append(ResourceInUserConstraint(ua, ra))
            for ra in self.multi_res_attrs:
                rv = r.attrs.get(ra)
                if isinstance(rv, frozenset) and rv and uv.issuperset(rv):
                    cons.append(SupersetConstraint(ua, ra))
        return Rule(sub_cond=sub_cond, res_cond=res_cond, acts={op}, cons=cons)

    # -- reliability / soundness -------------------------------------------------

    def _coverage(self, rule: Rule, universe_u_ids: FrozenSet[str], universe_r_ids: FrozenSet[str]) -> Set[Tuple[str, str]]:
        mu = self._user_idx.matched_ids(rule.sub_cond) & universe_u_ids
        mr = self._res_idx.matched_ids(rule.res_cond) & universe_r_ids
        if not mu or not mr:
            return set()
        if not rule.cons:
            return {(u, r) for u in mu for r in mr}
        covered = set()
        mu_entities = [(uid, self.users[uid]) for uid in mu]
        mr_entities = [(rid, self.resources[rid]) for rid in mr]
        for uid, u in mu_entities:
            for rid, r in mr_entities:
                if rule.matches_constraints(u, r):
                    covered.add((uid, rid))
        return covered

    def _reliability(self, rule: Rule, acl_pairs: Set[Tuple[str, str]], universe_u_ids: FrozenSet[str], universe_r_ids: FrozenSet[str]) -> float:
        covered = self._coverage(rule, universe_u_ids, universe_r_ids)
        if not covered:
            return 1.0
        correct = len(covered & acl_pairs)
        return correct / len(covered)

    def _generalize(self, rule: Rule, acl_pairs: Set[Tuple[str, str]], universe_u_ids: FrozenSet[str], universe_r_ids: FrozenSet[str]) -> Rule:
        changed = True
        while changed:
            changed = False
            for group in ("sub_cond", "res_cond", "cons"):
                for c in list(getattr(rule, group)):
                    trial = rule.clone()
                    getattr(trial, group).remove(c)
                    if self._reliability(trial, acl_pairs, universe_u_ids, universe_r_ids) >= self.config.min_reliability:
                        rule = trial
                        changed = True
        return rule

    # -- top-level mining --------------------------------------------------------

    def mine(self, log_grants: Sequence[Request]) -> Policy:
        by_op: Dict[str, Set[Tuple[str, str]]] = {}
        observed_uids: Set[str] = set()
        observed_rids: Set[str] = set()
        for uid, op, rid in log_grants:
            by_op.setdefault(op, set()).add((uid, rid))
            observed_uids.add(uid)
            observed_rids.add(rid)

        if self.config.universe == "observed":
            universe_u_ids: FrozenSet[str] = frozenset(observed_uids)
            universe_r_ids: FrozenSet[str] = frozenset(observed_rids)
        else:
            universe_u_ids = self._all_user_ids
            universe_r_ids = self._all_res_ids

        all_rules: List[Rule] = []
        for op, acl_pairs in by_op.items():
            uncovered = set(acl_pairs)
            guard = 0
            while uncovered:
                guard += 1
                if guard > self.config.max_rules_per_op:
                    raise RuntimeError(f"too many rules for op={op}; likely a bug")
                # min(), not next(iter(...)): `uncovered` is a set, whose iteration
                # order depends on the process's string hash seed (randomized per
                # process unless PYTHONHASHSEED is pinned). Picking the seed this way
                # made mining -- and therefore attack outcomes near a marginal
                # threshold -- vary between script invocations on identical inputs.
                uid, rid = min(uncovered)
                rule = self._specific_rule(uid, rid, op)
                rule = self._generalize(rule, acl_pairs, universe_u_ids, universe_r_ids)
                covered = self._coverage(rule, universe_u_ids, universe_r_ids)
                uncovered -= covered & acl_pairs
                all_rules.append(rule)

        merged = _compact(all_rules)
        return Policy(users=self.users, resources=self.resources, rules=merged)


def _compact(rules: List[Rule]) -> List[Rule]:
    """Merge rules that differ only in their (singleton) action set."""
    groups: Dict[Tuple[FrozenSet, FrozenSet, FrozenSet], Set[str]] = {}
    for rule in rules:
        key = (frozenset(rule.sub_cond), frozenset(rule.res_cond), frozenset(rule.cons))
        groups.setdefault(key, set()).update(rule.acts)
    out = []
    for (sub_cond, res_cond, cons), acts in groups.items():
        out.append(Rule(sub_cond=list(sub_cond), res_cond=list(res_cond), acts=acts, cons=list(cons)))
    return out


def mine_policy(
    users: Dict[str, Entity],
    resources: Dict[str, Entity],
    log_grants: Sequence[Request],
    min_reliability: float = 1.0,
    universe: str = "full",
) -> Policy:
    config = MinerConfig(min_reliability=min_reliability, universe=universe)
    miner = BottomUpABACMiner(users, resources, config)
    return miner.mine(log_grants)
