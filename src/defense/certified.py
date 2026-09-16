"""Problem 4 (Section 4.4, C5): certified legitimate-only robustness.

Claim: the strict bottom-up miner (miners/xu_stoller.py: min_reliability=1.0,
universe="full") is UNCONDITIONALLY robust to legitimate-only poisoning of any size,
for any target request whose subject and resource already exist in the policy's
enrolled population. This gives an infinite robustness radius (k = infinity) against
Problem 1's rule-generalization attack specifically -- it is not a general claim about
every attack in this threat model (Problems 2/3 use different learner families and
different mechanisms this certificate does not cover; a white-box or influence-based
variant of Problem 1 is also outside this argument if it could ever exploit something
other than the reliability/soundness check studied here).

Proof.
Let P* be ground truth, L the clean log, and Delta any legitimate-only poison set
(Section 3.1: every (q, l) in Delta has l = auth_{P*}(q) = 1, i.e. Delta only contains
TRUE grants). Assume the clean log itself is also faithful -- Section 1's own premise
is that the access log is treated as "a faithful record of legitimate behavior," so a
clean log entry is never a mislabelled or fabricated grant either. Then

    granted_pairs(L, op)     subset-of true_grants(P*, op)      for every op, and
    granted_pairs(Delta, op) subset-of true_grants(P*, op)      for every op (Sec. 3.1),

so for ANY Delta,

    granted_pairs(L u Delta, op) subset-of true_grants(P*, op).                     (*)

The strict miner (BottomUpABACMiner with min_reliability=1.0, universe="full") accepts
a candidate generalization G for operation op if and only if, checked over the FULL
enrolled population (every existing user x every existing resource, not just entities
seen in the log):

    coverage(G) subset-of granted_pairs(L u Delta, op).                             (**)

Let t* = (u_gain, op, r_gain) be a valid attack target (Section 3.3: auth_{P*}(t*) = 0),
with u_gain and r_gain both enrolled entities (they must be, for t* to denote a
meaningful escalation against an existing population -- see `is_valid_target`). Suppose
some candidate rule G considered during mining has t* in coverage(G). By (*),
t* not in granted_pairs(L u Delta, op) for ANY Delta (since t* not in true_grants(P*,
op) by definition of a valid target). So (**) fails for G, and G is never accepted --
regardless of |Delta|, its composition, or the adversary's knowledge level.

Because this holds independent of Delta's size, the certified robustness radius for
this decision is k = infinity: no legitimate-only poison set, of any size, can cause
the strict miner to grant t*. Note the argument is about the ACCEPTANCE CRITERION
(**), not the search strategy that walks toward candidate rules -- it applies to every
candidate the miner ever considers, not just whatever rule it ultimately outputs, so
there is no unconsidered path that could sneak past it.

Cost. This robustness is not free: Section 6 asks for "clean-log F1 of Mine-dagger."
Because (**) is checked against the log's OWN grants (not P* directly) and the miner
never widens a value set (only drops whole conjuncts -- see miners/common.py's
documented v1 simplification), a strict miner asked to mine a genuinely SPARSE clean
log (no adversary involved) will refuse many generalizations a human would consider
obviously safe, producing under-privilege (legitimate requests wrongly denied) and a
far less compact rule set than an optimal miner. Rhapsody's threshold relaxation exists
specifically to trade away this certificate for better utility on incomplete real
logs -- which is exactly the trade this module quantifies empirically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from abac.model import Policy
from attack.threat_model import is_valid_target

Request = Tuple[str, str, str]


@dataclass
class CertificateResult:
    target: Request
    preconditions_hold: bool
    robustness_radius: float  # math.inf if certified, else 0.0 (no guarantee made)

    @property
    def certified(self) -> bool:
        return self.preconditions_hold


def certify_target_robustness(policy_star: Policy, target: Request) -> CertificateResult:
    """Verify the certificate's preconditions for `target` under the strict miner
    (min_reliability=1.0, universe="full"): both the subject and resource must be
    enrolled entities, and the target must actually be denied under P* (a valid attack
    target per Section 3.3). If both hold, the strict miner is certified robust to ANY
    legitimate-only poison set for this decision, of any size -- see module docstring
    for the proof. This function only checks the preconditions; the guarantee itself
    follows from the miner's fixed acceptance rule, not from anything computed here.
    """
    uid, op, rid = target
    enrolled = uid in policy_star.users and rid in policy_star.resources
    denied = enrolled and is_valid_target(target, policy_star)
    holds = enrolled and denied
    return CertificateResult(
        target=target,
        preconditions_hold=holds,
        robustness_radius=math.inf if holds else 0.0,
    )
