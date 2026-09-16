"""Parser for the .abac policy language (Xu-Stoller / ABAC-Lab format).

Grammar (see datasets/1_abac_ground_truth/ABAC-Lab/DATASETS/README.md):

    userAttrib(uid, attr1=v1, attr2=v2, ...)
    resourceAttrib(rid, attr1=v1, attr2=v2, ...)
    rule(subCond; resCond; acts; cons)
    # comment

A value is an atomic string, or a set `{e1 e2 ...}` (space-separated elements).
subCond/resCond conjuncts (comma-separated): `attr [ {v1 v2 ...}` (in) or `attr ] value` (contains).
cons conjuncts (comma-separated): `aum > arm` | `aus [ arm` | `aum ] ars` | `aus = ars`.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .model import (
    Conjunct,
    Constraint,
    ContainsCond,
    Entity,
    EqualityConstraint,
    InCond,
    Policy,
    ResourceInUserConstraint,
    Rule,
    SupersetConstraint,
    UserInResourceConstraint,
)

_ATTRIB_RE = re.compile(r"^(userAttrib|resourceAttrib)\((.*)\)\s*$")
_RULE_RE = re.compile(r"^rule\((.*)\)\s*$")


def _split_top_level(s: str, sep: str) -> List[str]:
    """Split on `sep` but never inside a {...} set literal."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1].strip()
        return frozenset(inner.split()) if inner else frozenset()
    return raw


def _parse_attrib_line(body: str) -> Tuple[str, dict]:
    parts = _split_top_level(body, ",")
    eid = parts[0].strip()
    attrs = {}
    for kv in parts[1:]:
        kv = kv.strip()
        if not kv:
            continue
        key, _, val = kv.partition("=")
        attrs[key.strip()] = _parse_value(val)
    return eid, attrs


def _parse_conjunct(text: str) -> Conjunct:
    text = text.strip()
    if "[" in text:
        attr, _, rhs = text.partition("[")
        rhs = rhs.strip()
        assert rhs.startswith("{") and rhs.endswith("}"), f"malformed 'in' condition: {text!r}"
        values = frozenset(rhs[1:-1].split())
        return InCond(attr=attr.strip(), values=values)
    if "]" in text:
        attr, _, rhs = text.partition("]")
        return ContainsCond(attr=attr.strip(), value=rhs.strip())
    raise ValueError(f"unrecognized condition conjunct: {text!r}")


def _parse_cond_list(text: str) -> List[Conjunct]:
    text = text.strip()
    if not text:
        return []
    return [_parse_conjunct(c) for c in _split_top_level(text, ",") if c.strip()]


def _parse_constraint(text: str) -> Constraint:
    text = text.strip()
    for op, cls in ((">", SupersetConstraint), ("=", EqualityConstraint)):
        if op in text:
            lhs, _, rhs = text.partition(op)
            return cls(user_attr=lhs.strip(), res_attr=rhs.strip())
    if "[" in text:
        lhs, _, rhs = text.partition("[")
        return UserInResourceConstraint(user_attr=lhs.strip(), res_attr=rhs.strip())
    if "]" in text:
        lhs, _, rhs = text.partition("]")
        return ResourceInUserConstraint(user_attr=lhs.strip(), res_attr=rhs.strip())
    raise ValueError(f"unrecognized constraint conjunct: {text!r}")


def _parse_cons_list(text: str) -> List[Constraint]:
    text = text.strip()
    if not text:
        return []
    return [_parse_constraint(c) for c in _split_top_level(text, ",") if c.strip()]


def _parse_rule_line(body: str) -> Rule:
    fields = _split_top_level(body, ";")
    if len(fields) != 4:
        raise ValueError(f"expected 4 ';'-separated fields in rule, got {len(fields)}: {body!r}")
    sub_text, res_text, acts_text, cons_text = fields
    acts_text = acts_text.strip()
    assert acts_text.startswith("{") and acts_text.endswith("}"), f"malformed acts: {acts_text!r}"
    acts = set(acts_text[1:-1].split())
    return Rule(
        sub_cond=_parse_cond_list(sub_text),
        res_cond=_parse_cond_list(res_text),
        acts=acts,
        cons=_parse_cons_list(cons_text),
    )


def parse_abac_file(path: str) -> Policy:
    policy = Policy()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Join logical statements: a statement runs from `name(` to the matching `)`
    # possibly spanning multiple physical lines; strip full-line comments first.
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        lines.append(line)
    text = "\n".join(lines)

    statements = _extract_statements(text)
    for stmt in statements:
        m = _ATTRIB_RE.match(stmt)
        if m:
            kind, body = m.group(1), m.group(2)
            eid, attrs = _parse_attrib_line(body)
            entity = Entity(eid=eid, attrs=attrs)
            if kind == "userAttrib":
                entity.attrs["uid"] = eid
                policy.users[eid] = entity
            else:
                entity.attrs["rid"] = eid
                policy.resources[eid] = entity
            continue
        m = _RULE_RE.match(stmt)
        if m:
            policy.rules.append(_parse_rule_line(m.group(1)))
            continue
        raise ValueError(f"unrecognized statement: {stmt!r}")
    return policy


def _extract_statements(text: str) -> List[str]:
    """Split `text` into top-level `name(...)` statements, balancing parentheses."""
    statements = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r":
            i += 1
        if i >= n:
            break
        start = i
        depth = 0
        started = False
        while i < n:
            ch = text[i]
            if ch == "(":
                depth += 1
                started = True
            elif ch == ")":
                depth -= 1
            i += 1
            if started and depth == 0:
                break
        statements.append(text[start:i].strip())
    return statements
