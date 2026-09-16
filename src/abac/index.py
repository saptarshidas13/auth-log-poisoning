"""Attribute-value indices for fast rule matching over a population.

Turns matching a subCond/resCond against a whole population into a handful of
C-level set intersections instead of a per-entity Python attribute-dict lookup. This is
what makes both mining (miners/common.py) and ground-truth comparison (metrics.py)
tractable once a dataset reaches hundreds of users/resources.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Sequence, Set

from .model import Conjunct, ContainsCond, Entity, InCond


class AttrIndex:
    """attr -> value -> set of entity ids, for a chosen set of single-valued attributes."""

    def __init__(self, entities: Dict[str, Entity], single_valued_attrs: Set[str]):
        self.entities = entities
        self.all_ids: FrozenSet[str] = frozenset(entities)
        self._index: Dict[str, Dict[str, Set[str]]] = {a: {} for a in single_valued_attrs}
        for eid, e in entities.items():
            for a in single_valued_attrs:
                v = e.attrs.get(a)
                if v is not None and not isinstance(v, frozenset):
                    self._index[a].setdefault(v, set()).add(eid)

    def matched_ids(self, conjuncts: Sequence[Conjunct]) -> FrozenSet[str]:
        ids = self.all_ids
        for c in conjuncts:
            if isinstance(c, InCond):
                idx = self._index.get(c.attr, {})
                value_ids: Set[str] = set()
                for v in c.values:
                    value_ids |= idx.get(v, set())
                ids = ids & value_ids
            elif isinstance(c, ContainsCond):
                ids = frozenset(
                    eid for eid in ids
                    if isinstance(self.entities[eid].attrs.get(c.attr), frozenset)
                    and c.value in self.entities[eid].attrs[c.attr]
                )
            if not ids:
                return frozenset()
        return ids
