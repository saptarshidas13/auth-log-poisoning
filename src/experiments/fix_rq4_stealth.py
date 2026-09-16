"""Recompute RQ4 stealth metrics for all 5 datasets with the corrected
natural_variance_baseline (perturbs the SAME base_log the attack used, instead of a
differently-withheld log -- see analysis.py's docstring for why the original design
was a confound). Writes results/p1_rq4_stealth_corrected.{csv,json}.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abac.logs import build_partial_log, enumerate_full_log
from abac.parser import parse_abac_file
from attack import Adversary, find_near_miss_targets, greedy_rule_generalization_attack, natural_variance_baseline
from metrics import compare_to_ground_truth, policy_size
from miners import rhapsody
from pipeline import ExperimentRecord, ResultsWriter

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# (dataset, role_attr, tuned_tau, near_miss_index_or_uid) -- known-good configs from the
# already-completed RQ1 sweep, reconstructed here rather than re-searched.
CONFIGS = [
    ("university", "position", 0.5, None),
    ("healthcare", "position", 0.95, None),
    ("project-management", "position", 0.7, None),
    ("workforce", "position", 0.7, None),
    ("edocument", "role", 0.8, "user127"),
]


def run():
    writer = ResultsWriter("p1_rq4_stealth_corrected", RESULTS_DIR)
    for name, role_attr, tuned_tau, uid_filter in CONFIGS:
        path = os.path.join(DATASET_DIR, f"{name}.abac")
        policy_star = parse_abac_file(path)
        full_log = enumerate_full_log(policy_star)

        near_misses = find_near_miss_targets(policy_star, limit=20)
        if uid_filter:
            nm = next(n for n in near_misses if n.target[0] == uid_filter)
        else:
            nm = near_misses[0]
        target = nm.target
        role = policy_star.users[target[0]].attrs.get(role_attr)
        controlled = {uid for uid, e in policy_star.users.items() if e.attrs.get(role_attr) == role}
        adversary = Adversary(controlled_uids=frozenset(controlled), knowledge="grey_box", beta=0.9)
        base_log = build_partial_log(full_log, controlled, target[1], withhold_fraction=1.0, seed=0)

        trace = greedy_rule_generalization_attack(
            policy_star=policy_star,
            mine_fn=lambda u, r, log, tau=tuned_tau: rhapsody.mine(u, r, log, min_reliability=tau),
            base_log=base_log, target=target, adversary=adversary,
        )
        if not trace.success:
            print(f"{name}: reconstruction did not reproduce success at tau={tuned_tau}, skipping")
            continue

        baseline_stats = natural_variance_baseline(
            policy_star, lambda u, r, log: rhapsody.mine(u, r, log, min_reliability=tuned_tau),
            full_log, base_log, num_samples=8,
        )
        poisoned_log = base_log + trace.delta
        poisoned_policy = rhapsody.mine(policy_star.users, policy_star.resources, poisoned_log, min_reliability=tuned_tau)
        poisoned_report = compare_to_ground_truth(poisoned_policy, full_log)
        rule_count = policy_size(poisoned_policy)[0]

        rc_z = baseline_stats.z_score_rule_count(rule_count)
        op_z = baseline_stats.z_score_over_privilege(poisoned_report.over_privilege_count)
        print(f"{name} (tau={tuned_tau}, poison={len(trace.delta)}): "
              f"natural rule_count={baseline_stats.rule_count_mean:.1f}+/-{baseline_stats.rule_count_std:.2f}, "
              f"attacked={rule_count} (z={rc_z:.2f}); "
              f"natural over_priv={baseline_stats.over_privilege_mean:.1f}+/-{baseline_stats.over_privilege_std:.2f}, "
              f"attacked={poisoned_report.over_privilege_count} (z={op_z:.2f})")

        writer.add(ExperimentRecord(
            problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="grey_box",
            tau=tuned_tau, beta=0.9, poison_size=len(trace.delta), success=True,
            over_privilege_after=poisoned_report.over_privilege_count, rule_count_after=rule_count,
            extra={
                "rq": "RQ4_stealth_corrected",
                "natural_rule_count_mean": baseline_stats.rule_count_mean,
                "natural_rule_count_std": baseline_stats.rule_count_std,
                "natural_over_privilege_mean": baseline_stats.over_privilege_mean,
                "natural_over_privilege_std": baseline_stats.over_privilege_std,
                "rule_count_z": rc_z,
                "over_privilege_z": op_z,
            },
        ))
    writer.flush()


if __name__ == "__main__":
    run()
