"""Core ABAC data model: entities, conditions, rules, policies, and grant evaluation.

Implements the notation of Section 2 of the problem statement: users U, resources R,
operations O, a policy P mapping a request q = (u, o, r) to a decision in {0, 1}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Union

AttrValue = Union[str, FrozenSet[str]]


@dataclass
class Entity:
    """A user or resource: an id plus a bag of attribute name -> value.

    A value is either an atomic string (single-valued attribute) or a frozenset of
    strings (multi-valued attribute), matching the .abac grammar.
    """

    eid: str
    attrs: Dict[str, AttrValue] = field(default_factory=dict)


# --- Condition/constraint conjunct representations -------------------------------

@dataclass(frozen=True)
class InCond:
    """attr [ {v1 v2 ...}  -- single-valued entity attribute's value is in the set."""

    attr: str
    values: FrozenSet[str]


@dataclass(frozen=True)
class ContainsCond:
    """attr ] value  -- multi-valued entity attribute contains the atomic value."""

    attr: str
    value: str


Conjunct = Union[InCond, ContainsCond]


@dataclass(frozen=True)
class SupersetConstraint:
    """aum > arm : user's multi-valued attribute aum is a superset of resource's arm."""

    user_attr: str
    res_attr: str


@dataclass(frozen=True)
class UserInResourceConstraint:
    """aus [ arm : user's single-valued attribute value is in resource's multi-valued arm."""

    user_attr: str
    res_attr: str


@dataclass(frozen=True)
class ResourceInUserConstraint:
    """aum ] ars : user's multi-valued attribute contains resource's single-valued value."""

    user_attr: str
    res_attr: str


@dataclass(frozen=True)
class EqualityConstraint:
    """aus = ars : user's and resource's single-valued attributes are equal."""

    user_attr: str
    res_attr: str


Constraint = Union[
    SupersetConstraint, UserInResourceConstraint, ResourceInUserConstraint, EqualityConstraint
]


@dataclass
class Rule:
    """rule(subCond; resCond; acts; cons) -- a single ABAC rule."""

    sub_cond: List[Conjunct] = field(default_factory=list)
    res_cond: List[Conjunct] = field(default_factory=list)
    acts: Set[str] = field(default_factory=set)
    cons: List[Constraint] = field(default_factory=list)

    def clone(self) -> "Rule":
        return Rule(
            sub_cond=list(self.sub_cond),
            res_cond=list(self.res_cond),
            acts=set(self.acts),
            cons=list(self.cons),
        )

    def num_conjuncts(self) -> int:
        """Total conjuncts across subCond, resCond, and cons (a common miner quality term)."""
        return len(self.sub_cond) + len(self.res_cond) + len(self.cons)

    def matches_subject(self, u: Entity) -> bool:
        for c in self.sub_cond:
            if isinstance(c, InCond):
                val = u.attrs.get(c.attr)
                if val is None or isinstance(val, frozenset) or val not in c.values:
                    return False
            elif isinstance(c, ContainsCond):
                val = u.attrs.get(c.attr)
                if not isinstance(val, frozenset) or c.value not in val:
                    return False
        return True

    def matches_resource(self, r: Entity) -> bool:
        for c in self.res_cond:
            if isinstance(c, InCond):
                val = r.attrs.get(c.attr)
                if val is None or isinstance(val, frozenset) or val not in c.values:
                    return False
            elif isinstance(c, ContainsCond):
                val = r.attrs.get(c.attr)
                if not isinstance(val, frozenset) or c.value not in val:
                    return False
        return True

    def matches_constraints(self, u: Entity, r: Entity) -> bool:
        for c in self.cons:
            if isinstance(c, SupersetConstraint):
                uv = u.attrs.get(c.user_attr)
                rv = r.attrs.get(c.res_attr)
                uv = uv if isinstance(uv, frozenset) else frozenset()
                rv = rv if isinstance(rv, frozenset) else frozenset()
                if not uv.issuperset(rv):
                    return False
            elif isinstance(c, UserInResourceConstraint):
                uv = u.attrs.get(c.user_attr)
                rv = r.attrs.get(c.res_attr)
                if uv is None or not isinstance(rv, frozenset) or uv not in rv:
                    return False
            elif isinstance(c, ResourceInUserConstraint):
                uv = u.attrs.get(c.user_attr)
                rv = r.attrs.get(c.res_attr)
                if rv is None or not isinstance(uv, frozenset) or rv not in uv:
                    return False
            elif isinstance(c, EqualityConstraint):
                uv = u.attrs.get(c.user_attr)
                rv = r.attrs.get(c.res_attr)
                if uv is None or rv is None or uv != rv:
                    return False
        return True

    def grants(self, u: Entity, op: str, r: Entity) -> bool:
        return (
            op in self.acts
            and self.matches_subject(u)
            and self.matches_resource(r)
            and self.matches_constraints(u, r)
        )


@dataclass
class Policy:
    """A full ABAC policy P: users, resources, and a rule set. auth_P(q) via `decide`."""

    users: Dict[str, Entity] = field(default_factory=dict)
    resources: Dict[str, Entity] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)

    def actions(self) -> Set[str]:
        acts: Set[str] = set()
        for rule in self.rules:
            acts |= rule.acts
        return acts

    def decide(self, uid: str, op: str, rid: str) -> int:
        u = self.users[uid]
        r = self.resources[rid]
        for rule in self.rules:
            if rule.grants(u, op, r):
                return 1
        return 0

    def num_rules(self) -> int:
        return len(self.rules)

    def total_conjuncts(self) -> int:
        return sum(rule.num_conjuncts() for rule in self.rules)
