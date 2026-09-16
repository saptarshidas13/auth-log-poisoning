"""RQ3 (black-box transfer) and RQ4 (stealth) analysis for the rule-generalization attack.

RQ3: "Does the attack transfer black-box, i.e. does poison crafted without knowledge of
the miner still succeed?" `greedy_rule_generalization_attack` is itself adaptive -- it
re-mines after each candidate and stops on success, which implicitly assumes query
access to the deployed miner. `craft_fixed_poison` removes that assumption: it ranks
candidates by attribute overlap ONCE and takes the top `size`, with no feedback from
the target miner at all -- a genuinely non-adaptive, transferable poison set. `transfer_test`
then checks whether that one fixed Delta happens to succeed across a range of
reliability thresholds it was never tuned against.

RQ4: "Can the attack remain stealthy, holding rule count, reliability, and
clean-validation accuracy within normal ranges?" A bare "rule count went up by 3" is
meaningless without knowing how much rule count varies ANYWAY from ordinary log
sampling noise, with no adversary at all. `natural_variance_baseline` re-mines several
different random clean (unpoisoned) log samples to establish that baseline
distribution, so an attack's induced change can be judged against it instead of an
arbitrary threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Dict, List, Sequence, Tuple

from abac.logs import AccessLog
from abac.model import Entity, Policy
from metrics import compare_to_ground_truth, policy_size
from .rule_generalization import attribute_overlap_score
from .threat_model import Adversary, legitimate_action_space

Request = Tuple[str, str, str]
MineFn = Callable[[Dict[str, Entity], Dict[str, Entity], Sequence[Request]], Policy]


def craft_fixed_poison(
    policy_star: Policy,
    base_log: Sequence[Request],
    target: Request,
    adversary: Adversary,
    size: int,
) -> List[Request]:
    """The top `size` legitimate, not-yet-logged, attribute-closest candidates for
    `target` -- computed ONCE, with no query access to any specific miner. This is the
    genuinely black-box / non-adaptive poison set RQ3 asks about.
    """
    already_logged = set(base_log)
    action_space = legitimate_action_space(policy_star, adversary)
    op_gain = target[1]
    candidates = [q for q in action_space if q[1] == op_gain and q != target and q not in already_logged]
    candidates.sort(
        key=lambda q: attribute_overlap_score(policy_star.users, policy_star.resources, q, target),
        reverse=True,
    )
    return candidates[:size]


@dataclass
class TransferResult:
    tau: float
    baseline_already_granted: bool
    success: bool


def transfer_test(
    policy_star: Policy,
    mine_fn_family: Callable[[float], MineFn],
    base_log: Sequence[Request],
    delta: Sequence[Request],
    target: Request,
    taus: Sequence[float],
) -> List[TransferResult]:
    """Apply ONE fixed Delta (from craft_fixed_poison) against a family of miners
    (varying tau) it was never tuned against, and report where it still succeeds.
    """
    from .threat_model import apply_poison

    out = []
    poisoned_log = apply_poison(base_log, delta)
    for tau in taus:
        mine_fn = mine_fn_family(tau)
        baseline = mine_fn(policy_star.users, policy_star.resources, base_log)
        baseline_granted = baseline.decide(*target) == 1
        after = mine_fn(policy_star.users, policy_star.resources, poisoned_log)
        success = (after.decide(*target) == 1) and not baseline_granted
        out.append(TransferResult(tau=tau, baseline_already_granted=baseline_granted, success=success))
    return out


@dataclass
class StealthBaseline:
    rule_count_mean: float
    rule_count_std: float
    over_privilege_mean: float
    over_privilege_std: float
    samples: int

    def z_score_rule_count(self, observed: int) -> float:
        return 0.0 if self.rule_count_std == 0 else (observed - self.rule_count_mean) / self.rule_count_std

    def z_score_over_privilege(self, observed: int) -> float:
        return 0.0 if self.over_privilege_std == 0 else (observed - self.over_privilege_mean) / self.over_privilege_std


def natural_variance_baseline(
    policy_star: Policy,
    mine_fn: MineFn,
    full_log: AccessLog,
    base_log: Sequence[Request],
    num_samples: int = 10,
    perturbation: float = 0.02,
) -> StealthBaseline:
    """Re-mine `num_samples` slight perturbations of the EXACT `base_log` the attack
    used (no Delta), to establish how much rule count / over-privilege varies from
    ordinary observation noise alone, on a directly comparable footing.

    Deliberately perturbs `base_log` itself (drop a small random `perturbation`
    fraction of its own entries) rather than re-deriving a differently-withheld log:
    if the natural baseline used a different overall withhold_fraction than the
    attack's own base_log, the comparison would conflate the attack's effect with a
    systematic difference in how much data each log has -- exactly the kind of
    confound `sample_sparse_log` already burned us on (see
    [[feedback-poisoning-attack-experiment-design]] point 3). Perturbing the same log
    keeps overall completeness matched while still producing genuine seed-to-seed
    variance to compare against.
    """
    import random as _random

    rule_counts, over_privileges = [], []
    for seed in range(num_samples):
        rng = _random.Random(1000 + seed)
        n_drop = int(round(perturbation * len(base_log)))
        drop = set(rng.sample(range(len(base_log)), k=min(n_drop, len(base_log))))
        log = [q for i, q in enumerate(base_log) if i not in drop]
        mined = mine_fn(policy_star.users, policy_star.resources, log)
        report = compare_to_ground_truth(mined, full_log)
        rule_counts.append(policy_size(mined)[0])
        over_privileges.append(report.over_privilege_count)
    return StealthBaseline(
        rule_count_mean=mean(rule_counts),
        rule_count_std=pstdev(rule_counts),
        over_privilege_mean=mean(over_privileges),
        over_privilege_std=pstdev(over_privileges),
        samples=num_samples,
    )
