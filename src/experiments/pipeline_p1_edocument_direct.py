"""Targeted P1 sweep for edocument using the known-good target (found via manual
reconnaissance: user127/view/doc102, role=employee, 400-user group) directly, bypassing
the general pipeline's per-candidate reconnaissance (expensive at this dataset's scale).
Appends to results/p1_rule_generalization.{csv,json} if it already has other datasets'
rows, matching the same ExperimentRecord schema.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abac.logs import build_partial_log, enumerate_full_log
from abac.parser import parse_abac_file
from attack import (
    Adversary,
    craft_fixed_poison,
    find_near_miss_targets,
    greedy_rule_generalization_attack,
    natural_variance_baseline,
    transfer_test,
)
from metrics import compare_to_ground_truth, policy_size
from miners import rhapsody, xu_stoller
from pipeline import ExperimentRecord, ResultsWriter

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
TAUS = (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3)
BETA = 0.9


def run():
    name = "edocument"
    path = os.path.join(DATASET_DIR, f"{name}.abac")
    policy_star = parse_abac_file(path)
    full_log = enumerate_full_log(policy_star)

    near_misses = find_near_miss_targets(policy_star, limit=20)
    target = next(nm.target for nm in near_misses if nm.target[0] == "user127")
    nm = next(nm for nm in near_misses if nm.target == target)
    op_gain = target[1]
    role = policy_star.users[target[0]].attrs.get("role")
    controlled = {uid for uid, e in policy_star.users.items() if e.attrs.get("role") == role}
    adversary = Adversary(controlled_uids=frozenset(controlled), knowledge="grey_box", beta=BETA)
    base_log = build_partial_log(full_log, controlled, op_gain, withhold_fraction=1.0, seed=0)

    print(f"=== {name} (direct target) === target={target} ceiling={nm.baseline_reliability:.3f} "
          f"adversary_size={len(controlled)} base_log={len(base_log)}")

    records = []
    cheapest = None
    for tau in TAUS:
        t0 = time.time()
        trace = greedy_rule_generalization_attack(
            policy_star=policy_star,
            mine_fn=lambda u, r, log, tau=tau: rhapsody.mine(u, r, log, min_reliability=tau),
            base_log=base_log, target=target, adversary=adversary,
        )
        dt = time.time() - t0
        print(f"  RQ1 tau={tau}: success={trace.success} poison_size={len(trace.delta)} "
              f"baseline_granted={trace.baseline_already_granted} ({dt:.1f}s)")
        records.append(ExperimentRecord(
            problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="grey_box",
            tau=tau, beta=BETA, poison_size=len(trace.delta),
            poison_fraction=len(trace.delta) / len(base_log) if base_log else 0.0,
            success=trace.success, baseline_already_granted=trace.baseline_already_granted, seconds=dt,
            extra={"rq": "RQ1", "ceiling_reliability": nm.baseline_reliability},
        ))
        if trace.success and cheapest is None:
            cheapest = (tau, trace)

    t0 = time.time()
    control = greedy_rule_generalization_attack(
        policy_star=policy_star, mine_fn=lambda u, r, log: xu_stoller.mine(u, r, log),
        base_log=base_log, target=target, adversary=adversary,
    )
    dt = time.time() - t0
    print(f"  [control, strict Xu-Stoller]: success={control.success} poison_size={len(control.delta)} ({dt:.1f}s)")
    records.append(ExperimentRecord(
        problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="grey_box",
        tau=1.0, beta=BETA, poison_size=len(control.delta),
        poison_fraction=len(control.delta) / len(base_log) if base_log else 0.0,
        success=control.success, baseline_already_granted=control.baseline_already_granted, seconds=dt,
        extra={"rq": "control_strict_xu_stoller"},
    ))

    if cheapest is not None:
        tuned_tau, tuned_trace = cheapest
        fixed_delta = craft_fixed_poison(policy_star, base_log, target, adversary, size=len(tuned_trace.delta))
        transfer_results = transfer_test(
            policy_star, lambda tau: (lambda u, r, log, tau=tau: rhapsody.mine(u, r, log, min_reliability=tau)),
            base_log, fixed_delta, target, TAUS,
        )
        print(f"  RQ3 transfer (Delta size={len(fixed_delta)}, tuned for tau={tuned_tau}):")
        for tr in transfer_results:
            print(f"    tau={tr.tau}: success={tr.success} baseline_granted={tr.baseline_already_granted}")
            records.append(ExperimentRecord(
                problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="black_box",
                tau=tr.tau, beta=BETA, poison_size=len(fixed_delta),
                poison_fraction=len(fixed_delta) / len(base_log) if base_log else 0.0,
                success=tr.success, baseline_already_granted=tr.baseline_already_granted,
                extra={"rq": "RQ3_transfer", "tuned_for_tau": tuned_tau},
            ))

        baseline_stats = natural_variance_baseline(
            policy_star, lambda u, r, log: rhapsody.mine(u, r, log, min_reliability=tuned_tau),
            full_log, base_log, num_samples=5,
        )
        poisoned_log = base_log + tuned_trace.delta
        poisoned_policy = rhapsody.mine(policy_star.users, policy_star.resources, poisoned_log, min_reliability=tuned_tau)
        poisoned_report = compare_to_ground_truth(poisoned_policy, full_log)
        rule_count = policy_size(poisoned_policy)[0]
        print(f"  RQ4 stealth (tau={tuned_tau}): natural rule_count={baseline_stats.rule_count_mean:.1f}"
              f"+/-{baseline_stats.rule_count_std:.1f}, attacked={rule_count} "
              f"(z={baseline_stats.z_score_rule_count(rule_count):.2f}); "
              f"natural over_priv={baseline_stats.over_privilege_mean:.1f}"
              f"+/-{baseline_stats.over_privilege_std:.1f}, attacked={poisoned_report.over_privilege_count} "
              f"(z={baseline_stats.z_score_over_privilege(poisoned_report.over_privilege_count):.2f})")
        records.append(ExperimentRecord(
            problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="grey_box",
            tau=tuned_tau, beta=BETA, poison_size=len(tuned_trace.delta), success=True,
            over_privilege_after=poisoned_report.over_privilege_count, rule_count_after=rule_count,
            extra={
                "rq": "RQ4_stealth",
                "natural_rule_count_mean": baseline_stats.rule_count_mean,
                "natural_rule_count_std": baseline_stats.rule_count_std,
                "natural_over_privilege_mean": baseline_stats.over_privilege_mean,
                "natural_over_privilege_std": baseline_stats.over_privilege_std,
                "rule_count_z": baseline_stats.z_score_rule_count(rule_count),
                "over_privilege_z": baseline_stats.z_score_over_privilege(poisoned_report.over_privilege_count),
            },
        ))

    # Write standalone; merged into the main results file by the caller if needed.
    writer = ResultsWriter("p1_edocument_direct", RESULTS_DIR)
    writer.records = records
    writer.flush()


if __name__ == "__main__":
    run()
