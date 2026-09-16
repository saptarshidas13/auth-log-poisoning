"""Problem 1 (Section 4.1): rule-generalization poisoning against bottom-up ABAC rule miners.

Mechanism exploited: our miners (miners/common.py) accept a candidate generalization
(dropping one conjunct from a rule) once its RELIABILITY -- the fraction of everything it
would grant that is actually attested as a true grant in the (possibly sparse) log --
clears a threshold tau. The denominator (total coverage of the relaxed rule) is fixed by
which attributes remain and is outside the adversary's control. The NUMERATOR (grants
inside that coverage attested in the log) is exactly what a legitimate-only poison
record can inflate -- *provided the poisoned record itself falls inside that same
broadened coverage*. Enough such "attribute-neighbor" records push reliability over tau,
the generalization is accepted, and the target t* -- which was already inside that same
coverage -- gets granted as a side effect, without ever being poisoned directly.

Two pieces:

  find_near_miss_targets() -- surfaces (rule, dropped-conjunct, target) opportunities:
      requests exactly one relaxation away from an existing ground-truth rule. This
      gives principled, reproducible attack targets instead of hand-picked ones, ranked
      by how close the relaxation already is to being sound (least poisoning needed).

  greedy_rule_generalization_attack() -- Section 4.1's stated fallback strategy: rank the
      adversary's own legitimate action space by attribute overlap with t*, and greedily
      add top-ranked, not-yet-logged records until the target flips or budget runs out.

Note on why the base log must be *sparse*: our reliability check is set-based
(|covered n acl_pairs| / |covered|). If the log already contains every true grant for an
operation, acl_pairs cannot grow -- adding an already-logged record is a no-op. The
attack only has purchase because real logs are incomplete: the adversary's "poisoning"
is the act of newly exercising (and thereby newly logging) permissions they already
held but simply hadn't used yet -- exactly Section 1's framing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from abac.index import AttrIndex
from abac.model import Entity, Policy, Rule
from abac.schema import single_valued_attrs
from .threat_model import Adversary, apply_poison, legitimate_action_space

Request = Tuple[str, str, str]
MineFn = Callable[[Dict[str, Entity], Dict[str, Entity], Sequence[Request]], Policy]


def _rule_coverage(rule: Rule, u_index: AttrIndex, r_index: AttrIndex, users, resources, universe_u_ids, universe_r_ids):
    mu = u_index.matched_ids(rule.sub_cond) & universe_u_ids
    mr = r_index.matched_ids(rule.res_cond) & universe_r_ids
    if not mu or not mr:
        return set()
    if not rule.cons:
        return {(u, r) for u in mu for r in mr}
    covered = set()
    for uid in mu:
        u = users[uid]
        for rid in mr:
            r = resources[rid]
            if rule.matches_constraints(u, r):
                covered.add((uid, rid))
    return covered


@dataclass
class NearMissTarget:
    target: Request
    source_rule: Rule
    dropped_conjunct: object
    baseline_reliability: float  # reliability of the relaxed rule against full ground truth


def find_near_miss_targets(policy_star: Policy, limit: int = 20) -> List[NearMissTarget]:
    """Requests one conjunct-drop away from an existing ground-truth rule, ranked by
    baseline reliability (highest first) so the "easiest" opportunities -- the ones
    needing the least poisoning to tip a threshold -- come first.
    """
    user_attrs = single_valued_attrs(policy_star.users, exclude=set())
    res_attrs = single_valued_attrs(policy_star.resources, exclude=set())
    u_index = AttrIndex(policy_star.users, user_attrs)
    r_index = AttrIndex(policy_star.resources, res_attrs)
    all_u, all_r = u_index.all_ids, r_index.all_ids

    true_grants = set()
    for rule in policy_star.rules:
        cov = _rule_coverage(rule, u_index, r_index, policy_star.users, policy_star.resources, all_u, all_r)
        for op in rule.acts:
            for uid, rid in cov:
                true_grants.add((uid, op, rid))

    best_for_target: Dict[Request, NearMissTarget] = {}
    for rule in policy_star.rules:
        for group in ("sub_cond", "res_cond", "cons"):
            for c in getattr(rule, group):
                relaxed = rule.clone()
                getattr(relaxed, group).remove(c)
                cov = _rule_coverage(relaxed, u_index, r_index, policy_star.users, policy_star.resources, all_u, all_r)
                if not cov:
                    continue
                correct, wrong = 0, []
                for uid, rid in cov:
                    for op in rule.acts:
                        req = (uid, op, rid)
                        if req in true_grants:
                            correct += 1
                        else:
                            wrong.append(req)
                total = correct + len(wrong)
                reliability = correct / total if total else 1.0
                if reliability >= 1.0:
                    continue  # relaxation is already fully sound, not a real "near miss"
                for req in wrong:
                    prev = best_for_target.get(req)
                    if prev is None or reliability > prev.baseline_reliability:
                        best_for_target[req] = NearMissTarget(req, rule, c, reliability)

    # Secondary sort key on the target tuple itself makes tie-breaking deterministic
    # across runs -- without it, ties resolve by set/dict iteration order, which
    # depends on Python's per-process string hash randomization (PYTHONHASHSEED).
    out = sorted(best_for_target.values(), key=lambda nm: (-nm.baseline_reliability, nm.target))
    return out[:limit]


def attribute_overlap_score(users: Dict[str, Entity], resources: Dict[str, Entity], candidate: Request, target: Request) -> int:
    """How many (name, value) pairs candidate's (user, resource) share with target's --
    the ranking heuristic Section 4.1 names as the fallback for miners without an
    explicit objective to derive poison from directly."""
    u_c, op_c, r_c = candidate
    u_t, op_t, r_t = target
    if op_c != op_t:
        return -1
    ua, ra = users[u_c].attrs, resources[r_c].attrs
    ub, rb = users[u_t].attrs, resources[r_t].attrs
    score = 0
    for k in set(ua) & set(ub):
        if not isinstance(ua[k], frozenset) and ua[k] == ub[k]:
            score += 1
    for k in set(ra) & set(rb):
        if not isinstance(ra[k], frozenset) and ra[k] == rb[k]:
            score += 1
    return score


@dataclass
class GreedyAttackTrace:
    delta: List[Request]
    history: List[Tuple[int, bool]]  # (|Delta| so far, granted at that size)
    success: bool  # True iff Delta is what caused the flip (baseline was NOT already granting it)
    baseline_already_granted: bool  # target granted by mine_fn(base_log) alone, before any poisoning


def greedy_rule_generalization_attack(
    policy_star: Policy,
    mine_fn: MineFn,
    base_log: Sequence[Request],
    target: Request,
    adversary: Adversary,
    step_size: int = None,
) -> GreedyAttackTrace:
    """Find a near-minimal poison size via exponential probing + binary search over the
    attribute-overlap-ranked candidate list, rather than linear stepping.

    Linear stepping costs O(budget / step_size) re-mining calls -- fine on the small
    ABAC datasets (sub-second mining), but on workforce/edocument-scale logs (5-60s per
    call) a large candidate pool made this run for tens of minutes per tau (see
    [[feedback-poisoning-attack-experiment-design]]). Exponential probe (1, 2, 4, 8, ...)
    followed by binary search between the last failure and first success costs
    O(log(budget)) calls and finds an equally tight (off by at most 1) minimal size.

    `step_size` is accepted for backward compatibility with earlier experiment scripts
    but is otherwise unused -- the search is now size-adaptive regardless of its value.
    """
    # Baseline check first: a target already granted by natural sparse-log noise, with
    # zero poisoning, is not something this attack caused -- crediting it to the
    # adversary would conflate this paper's threat (Section 1) with the benign i.i.d.
    # label-noise setting it explicitly distinguishes itself from (Section 7,
    # Positioning). Only a flip strictly caused by Delta counts as `success`.
    baseline_mined = mine_fn(policy_star.users, policy_star.resources, base_log)
    if baseline_mined.decide(*target) == 1:
        return GreedyAttackTrace(delta=[], history=[(0, True)], success=False, baseline_already_granted=True)

    already_logged = set(base_log)
    action_space = legitimate_action_space(policy_star, adversary)
    op_gain = target[1]
    candidates = [q for q in action_space if q[1] == op_gain and q != target and q not in already_logged]
    candidates.sort(
        key=lambda q: attribute_overlap_score(policy_star.users, policy_star.resources, q, target),
        reverse=True,
    )

    max_size = min(int(adversary.beta * len(base_log)), len(candidates))
    history: List[Tuple[int, bool]] = [(0, False)]

    def granted_at(size: int) -> bool:
        delta = candidates[:size]
        poisoned = apply_poison(base_log, delta)
        mined = mine_fn(policy_star.users, policy_star.resources, poisoned)
        result = mined.decide(*target) == 1
        history.append((size, result))
        return result

    if max_size <= 0:
        return GreedyAttackTrace(delta=[], history=history, success=False, baseline_already_granted=False)

    # Exponential probe for a failing/succeeding bracket.
    lo, hi = 0, 0
    size = 1
    found_success_at = None
    while size <= max_size:
        if granted_at(size):
            found_success_at = size
            hi = size
            break
        lo = size
        size *= 2
    else:
        size = None

    if found_success_at is None:
        if lo < max_size and granted_at(max_size):
            found_success_at = max_size
            hi = max_size
        else:
            best_size = lo if lo > 0 else max_size
            return GreedyAttackTrace(
                delta=candidates[:best_size], history=history, success=False, baseline_already_granted=False,
            )

    # Binary search in (lo, hi] for the minimal successful size.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if granted_at(mid):
            hi = mid
        else:
            lo = mid

    return GreedyAttackTrace(delta=candidates[:hi], history=history, success=True, baseline_already_granted=False)
