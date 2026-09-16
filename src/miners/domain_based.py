"""Domain-based policy mining (Zhang & Fong, CODASPY 2024 + SACMAT 2026).

This is "the SACMAT 2026 MDL noisy-log miner" the problem statement (Section 6.2)
names as the noise-robust baseline to beat: Si Zhang, Philip W. L. Fong, "Mining
Domain-Based Policies from Massive and Noisy Access Logs," SACMAT 2026. We
reimplement the algorithm as specified in the verified full text: the domain-based
policy model (their Section 2.1), the MDL two-part code (their Section 4.1-4.2), and
the AutoPart-style mining algorithm (their Section 4.3, Algorithms 2-3).

Their GPU acceleration is a scalability engineering contribution, not part of the
algorithm's decision logic -- a CPU/numpy implementation computes the same policy,
just slower, which is fine at our scale (hundreds of entities vs. their 10k-100k GPU
benchmarks).

Model. A domain-based policy P = (H, pi) groups entities (both subjects and objects --
their model treats them uniformly, IoT-style) into domains; pi: entities -> domains,
H is an edge-labelled digraph over domains. auth_P(u, a, v) = 1 iff (pi(u), a, pi(v))
in E(H). We combine our users and resources into one entity universe (id-prefixed
'u:'/'r:' to guarantee no collision) to match this model exactly.

Known, documented simplification: their Algorithm 2 (AUTOPART-GPU) has an inner
local-search reassignment loop described only at a high level ("up to n/10 entities
per iteration, based on how promising they are"). We omit it and rely on the primary
mechanism -- Hamming-distance-guided divisive splitting (Algorithm 3), accepted only
when it reduces the total MDL score -- which is the paper's stated core contribution;
the inner loop is described as a refinement on top of it. This is analogous to our
own miners/common.py's documented v1 simplifications elsewhere in this codebase.

Adaptation for our threat model: their training-set model assumes examples are
labelled with the TRUE ground-truth decision (which can be 0 or 1) at generation
time (Section 5.2, "we generate labelled examples ((u,a,v), l)"). Our own access
logs (abac/logs.py) only ever record grants. `sample_observed_denials` supplies a
fixed set of true denials from the ground-truth policy to play the role of "denied
requests the system also logs" -- something a legitimate-only adversary cannot
fabricate, remove, or bias (Section 3.1 restricts Delta to positive insertions only).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from abac.model import Policy

Request = Tuple[str, str, str]


def _uid_key(uid: str) -> str:
    return f"u:{uid}"


def _rid_key(rid: str) -> str:
    return f"r:{rid}"


@dataclass
class DomainBasedPolicy:
    entity_index: Dict[str, int]
    action_index: Dict[str, int]
    domain_of: np.ndarray  # (n,) int
    H: np.ndarray  # (M, k, M) bool

    def decide(self, uid: str, op: str, rid: str) -> int:
        ui = self.entity_index.get(_uid_key(uid))
        ri = self.entity_index.get(_rid_key(rid))
        ai = self.action_index.get(op)
        if ui is None or ri is None or ai is None:
            return 0
        c, d = int(self.domain_of[ui]), int(self.domain_of[ri])
        return int(self.H[c, ai, d])

    def num_rules(self) -> int:
        return int(self.H.sum())

    def num_domains(self) -> int:
        return int(self.domain_of.max()) + 1 if self.domain_of.size else 0


class Universe:
    """Fixed entity/action indexing shared by all tensors built for one dataset."""

    def __init__(self, users: Sequence[str], resources: Sequence[str], actions: Sequence[str]):
        ids = [_uid_key(u) for u in sorted(users)] + [_rid_key(r) for r in sorted(resources)]
        self.entity_index: Dict[str, int] = {e: i for i, e in enumerate(ids)}
        self.n = len(ids)
        self.action_index: Dict[str, int] = {a: i for i, a in enumerate(sorted(actions))}
        self.k = len(self.action_index)

    def build_tensors(self, pos: Sequence[Request], neg: Sequence[Request]):
        T1 = np.zeros((self.n, self.k, self.n), dtype=np.float64)
        T0 = np.zeros((self.n, self.k, self.n), dtype=np.float64)
        for uid, op, rid in pos:
            ui, ri, ai = self.entity_index[_uid_key(uid)], self.entity_index[_rid_key(rid)], self.action_index[op]
            T1[ui, ai, ri] += 1
        for uid, op, rid in neg:
            ui, ri, ai = self.entity_index[_uid_key(uid)], self.entity_index[_rid_key(rid)], self.action_index[op]
            T0[ui, ai, ri] += 1
        return T1, T0


def sample_observed_denials(policy_star: Policy, count: int, seed: int = 0) -> List[Request]:
    """A fixed set of true denials the defender's log also records -- the
    legitimate-only adversary cannot fabricate, remove, or bias these."""
    rng = random.Random(seed)
    uids = sorted(policy_star.users)
    rids = sorted(policy_star.resources)
    ops = sorted(policy_star.actions())
    denials: List[Request] = []
    seen = set()
    attempts, max_attempts = 0, count * 50 + 2000
    while len(denials) < count and attempts < max_attempts:
        attempts += 1
        uid, op, rid = rng.choice(uids), rng.choice(ops), rng.choice(rids)
        if (uid, op, rid) in seen:
            continue
        seen.add((uid, op, rid))
        if policy_star.decide(uid, op, rid) == 0:
            denials.append((uid, op, rid))
    return denials


# --- MDL scoring (Section 4.1-4.2) -----------------------------------------------

def _log_star_2(x: float) -> float:
    if x <= 1:
        return 0.0
    total, val = 0.0, np.log2(x)
    while val > 0:
        total += val
        val = np.log2(val) if val > 1 else 0.0
    return total


def model_bits(M: int, n: int, k: int) -> float:
    """L(P) = log*_2(M) + n*log2(M) + k*M^2 (Section 4.2)."""
    return _log_star_2(M) + n * np.log2(max(M, 1)) + k * (M ** 2)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    ent = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    ent[mask] = -p[mask] * np.log2(p[mask]) - (1 - p[mask]) * np.log2(1 - p[mask])
    return ent


def _indicator(domain_of: np.ndarray, M: int) -> np.ndarray:
    n = domain_of.shape[0]
    ind = np.zeros((n, M), dtype=np.float64)
    ind[np.arange(n), domain_of] = 1.0
    return ind


def data_bits(T1: np.ndarray, T0: np.ndarray, domain_of: np.ndarray, M: int) -> float:
    """L(S|P) = sum_{c,a,d} W(c,a,d) * H(p(c,a,d)) (Section 4.2)."""
    k = T1.shape[1]
    ind = _indicator(domain_of, M)
    total = 0.0
    for a in range(k):
        W1 = ind.T @ T1[:, a, :] @ ind
        W0 = ind.T @ T0[:, a, :] @ ind
        W = W1 + W0
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(W > 0, W1 / np.maximum(W, 1e-12), 0.0)
        total += float(np.sum(W * _binary_entropy(p)))
    return total


def total_mdl(T1: np.ndarray, T0: np.ndarray, domain_of: np.ndarray, M: int) -> float:
    n = T1.shape[0]
    k = T1.shape[1]
    return model_bits(M, n, k) + data_bits(T1, T0, domain_of, M)


# --- Algorithm 3: HammingSplit -----------------------------------------------

def _hamming(T1: np.ndarray, T0: np.ndarray, u: int, v: int) -> int:
    row = np.sum((T1[u] > 0) & (T0[v] > 0)) + np.sum((T0[u] > 0) & (T1[v] > 0))
    col = np.sum((T1[:, :, u] > 0) & (T0[:, :, v] > 0)) + np.sum((T0[:, :, u] > 0) & (T1[:, :, v] > 0))
    return int(row + col)


def hamming_split(T1: np.ndarray, T0: np.ndarray, members: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Algorithm 3: sample up to 50 members, find the farthest pair as seeds,
    assign each member to whichever seed it's closer to. Returns the subset
    assigned to the "new" domain (empty if no beneficial split is found)."""
    if len(members) < 2:
        return np.array([], dtype=int)
    sample_size = min(50, len(members))
    sample = rng.choice(members, size=sample_size, replace=False)

    best = (-1, None, None)
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            d = _hamming(T1, T0, sample[i], sample[j])
            if d > best[0]:
                best = (d, sample[i], sample[j])
    if best[0] <= 0:
        return np.array([], dtype=int)
    s1, s2 = best[1], best[2]

    new_domain = [u for u in members if _hamming(T1, T0, u, s1) > _hamming(T1, T0, u, s2)]
    return np.array(new_domain, dtype=int)


# --- Algorithm 2: AutoPart (core mechanism; inner reassignment loop omitted, --
# --- see module docstring) ---------------------------------------------------
#
# Deviation from the paper's pseudocode, documented: Algorithm 2 attempts to split
# EVERY current domain in one round and accepts/rejects the WHOLE round based on
# aggregate MDL. Empirically (on our smaller-n, moderate-k ABAC datasets) this batch
# acceptance lets one domain's unfavorable split veto another's clearly beneficial
# one, so the search stalls at a tiny M. We instead accept splits ONE AT A TIME,
# greedily taking whichever single candidate split improves total MDL the most each
# round -- standard greedy divisive clustering, and a more robust way to reach the
# same kind of locally-optimal partition their inner reassignment loop (which we
# omitted) would otherwise help find.

def autopart_mine(T1: np.ndarray, T0: np.ndarray, seed: int = 0, max_splits: int = 500) -> Tuple[np.ndarray, int]:
    n, k, _ = T1.shape
    rng = np.random.RandomState(seed)
    domain_of = np.zeros(n, dtype=int)
    M = 1
    L_best = total_mdl(T1, T0, domain_of, M)

    for _ in range(max_splits):
        best_candidate = None  # (L_new, domain_of_candidate, M_candidate)
        for d in range(M):
            members = np.where(domain_of == d)[0]
            if len(members) < 2:
                continue
            new_members = hamming_split(T1, T0, members, rng)
            if not (0 < len(new_members) < len(members)):
                continue
            candidate = domain_of.copy()
            candidate[new_members] = M  # tentatively the next free domain id
            L_candidate = total_mdl(T1, T0, candidate, M + 1)
            if best_candidate is None or L_candidate < best_candidate[0]:
                best_candidate = (L_candidate, candidate, M + 1)
        if best_candidate is None or best_candidate[0] >= L_best:
            break
        L_best, domain_of, M = best_candidate

    return domain_of, M


def build_H(T1: np.ndarray, T0: np.ndarray, domain_of: np.ndarray, M: int) -> np.ndarray:
    """H[c,a,d] = 1 iff W1[c,a,d] / W[c,a,d] > 0.5 (Algorithm 2, lines 17-20)."""
    k = T1.shape[1]
    ind = _indicator(domain_of, M)
    H = np.zeros((M, k, M), dtype=bool)
    for a in range(k):
        W1 = ind.T @ T1[:, a, :] @ ind
        W0 = ind.T @ T0[:, a, :] @ ind
        W = W1 + W0
        with np.errstate(divide="ignore", invalid="ignore"):
            H[:, a, :] = np.where(W > 0, (W1 / np.maximum(W, 1e-12)) > 0.5, False)
    return H


# --- Section 3: clean-setting graph-coloring pipeline ------------------------
#
# Guarantees consistency with the training set by construction (their Observation
# 3.2(2): a proper coloring never puts incompatible entities in the same domain).
# We use a simple greedy (Welsh-Powell-style) coloring rather than their
# GPU-parallel IPGC heuristic -- both are heuristic colorings of the SAME conflict
# graph, so they compute policies with the same certification property (any proper
# coloring gives a consistent policy) even though the number of domains may differ.

def build_conflict_graph(T1: np.ndarray, T0: np.ndarray) -> np.ndarray:
    """Undirected incompatibility graph (Definition 3.1(2)-(3)): u,v conflict iff
    they are row- or column-incompatible for some action."""
    n, k, _ = T1.shape
    conflict = np.zeros((n, n), dtype=bool)
    for a in range(k):
        pos = (T1[:, a, :] > 0).astype(np.int32)
        neg = (T0[:, a, :] > 0).astype(np.int32)
        row_overlap = pos @ neg.T  # [u,v] = #w with T1[u,a,w] and T0[v,a,w]
        conflict |= (row_overlap > 0) | (row_overlap > 0).T
        col_overlap = pos.T @ neg  # [u,v] = #w with T1[w,a,u] and T0[w,a,v]
        conflict |= (col_overlap > 0) | (col_overlap > 0).T
    np.fill_diagonal(conflict, False)
    return conflict


def greedy_coloring(conflict: np.ndarray) -> np.ndarray:
    """Welsh-Powell-style greedy coloring: process vertices by descending degree,
    assign the smallest color not used by an already-colored neighbor."""
    n = conflict.shape[0]
    degree = conflict.sum(axis=1)
    order = np.argsort(-degree)
    color = np.full(n, -1, dtype=int)
    for u in order:
        neighbor_colors = set(color[conflict[u]].tolist())
        neighbor_colors.discard(-1)
        c = 0
        while c in neighbor_colors:
            c += 1
        color[u] = c
    return color


def build_H_clean(T1: np.ndarray, color: np.ndarray, M: int) -> np.ndarray:
    """H(a) contains (c,d) iff SOME positive example (u,a,v) has color(u)=c,
    color(v)=d -- deny-by-default (Definition 3.1(4)/Observation 3.2(6))."""
    k = T1.shape[1]
    ind = _indicator(color, M)
    H = np.zeros((M, k, M), dtype=bool)
    for a in range(k):
        W1 = ind.T @ (T1[:, a, :] > 0).astype(np.float64) @ ind
        H[:, a, :] = W1 > 0
    return H


def mine_clean(
    users: Dict[str, object],
    resources: Dict[str, object],
    log_grants: Sequence[Request],
    observed_denials: Sequence[Request],
    actions: Sequence[str],
) -> DomainBasedPolicy:
    """Mine a domain-based policy via the graph-coloring pipeline (clean setting,
    Section 3) -- guaranteed consistent with (log_grants, observed_denials) by
    construction, analogous to our own miners' strict Xu-Stoller variant."""
    universe = Universe(list(users), list(resources), actions)
    T1, T0 = universe.build_tensors(log_grants, observed_denials)
    conflict = build_conflict_graph(T1, T0)
    color = greedy_coloring(conflict)
    M = int(color.max()) + 1 if color.size else 0
    H = build_H_clean(T1, color, M)
    return DomainBasedPolicy(entity_index=universe.entity_index, action_index=universe.action_index, domain_of=color, H=H)


def mine(
    users: Dict[str, object],
    resources: Dict[str, object],
    log_grants: Sequence[Request],
    observed_denials: Sequence[Request],
    actions: Sequence[str],
    seed: int = 0,
) -> DomainBasedPolicy:
    """Mine a domain-based policy via the MDL/AutoPart algorithm (noisy setting)."""
    universe = Universe(list(users), list(resources), actions)
    T1, T0 = universe.build_tensors(log_grants, observed_denials)
    domain_of, M = autopart_mine(T1, T0, seed=seed)
    H = build_H(T1, T0, domain_of, M)
    return DomainBasedPolicy(
        entity_index=universe.entity_index, action_index=universe.action_index, domain_of=domain_of, H=H,
    )
