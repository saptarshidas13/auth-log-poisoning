"""Infer single-valued vs multi-valued attribute schemas from parsed entities.

Returns sorted lists, not sets: these results get iterated to build candidate rule
conjuncts (miners/common.py), and a plain set's iteration order depends on the
process's string hash seed (randomized per-process unless PYTHONHASHSEED is pinned).
That made the greedy miner's conjunct-drop order -- and therefore which locally-optimal
rule it lands on -- vary between script invocations on identical inputs. Sorting fixes
it; membership tests (`a in ...`) still work fine on a list at this tiny scale (a
handful of attribute names).
"""
from __future__ import annotations

from typing import Dict, List, Set

from .model import Entity


def infer_attr_types(entities: Dict[str, Entity]) -> Dict[str, bool]:
    """Return {attr_name: is_multi_valued} by scanning all entities' attribute values."""
    is_multi: Dict[str, bool] = {}
    for e in entities.values():
        for attr, val in e.attrs.items():
            if isinstance(val, frozenset):
                is_multi[attr] = True
            else:
                is_multi.setdefault(attr, False)
    return is_multi


def single_valued_attrs(entities: Dict[str, Entity], exclude: Set[str]) -> List[str]:
    types = infer_attr_types(entities)
    return sorted(a for a, multi in types.items() if not multi and a not in exclude)


def multi_valued_attrs(entities: Dict[str, Entity], exclude: Set[str]) -> List[str]:
    types = infer_attr_types(entities)
    return sorted(a for a, multi in types.items() if multi and a not in exclude)
