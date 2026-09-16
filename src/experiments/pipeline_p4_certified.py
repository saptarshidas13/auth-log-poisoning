"""Structured Problem 4 sweep: certificate stress test (unlimited budget) and
utility-cost tradeoff curve, across all 5 ABAC-Lab datasets. Writes
results/p4_certificate.{csv,json}.
"""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abac.logs import build_partial_log, enumerate_full_log
from abac.parser import parse_abac_file
from attack import Adversary, find_near_miss_targets, greedy_rule_generalization_attack
from defense import certify_target_robustness
from metrics import compare_to_ground_truth
from miners import rhapsody, xu_stoller
from pipeline import ExperimentRecord, ResultsWriter

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DATASETS = ["university", "healthcare", "project-management", "workforce", "edocument"]
ROLE_ATTR_BY_DATASET = {
    "university": "position", "healthcare": "position", "project-management": "position",
    "workforce": "position", "edocument": "role",
}
UTILITY_TAUS = (0.9, 0.7, 0.5, 0.3)


def stress_test(writer: ResultsWriter, name: str, num_targets: int = 3):
    path = os.path.join(DATASET_DIR, f"{name}.abac")
    policy_star = parse_abac_file(path)
    full_log = enumerate_full_log(policy_star)
    role_attr = ROLE_ATTR_BY_DATASET[name]
    near_misses = find_near_miss_targets(policy_star, limit=num_targets)

    print(f"\n=== {name} certificate stress test ===")
    for nm in near_misses:
        target = nm.target
        cert = certify_target_robustness(policy_star, target)
        role = policy_star.users[target[0]].attrs.get(role_attr)
        controlled = {uid for uid, e in policy_star.users.items() if e.attrs.get(role_attr) == role}
        adversary = Adversary(controlled_uids=frozenset(controlled), knowledge="white_box", beta=1.0)
        base_log = build_partial_log(full_log, controlled, target[1], withhold_fraction=1.0, seed=0)

        t0 = time.time()
        trace = greedy_rule_generalization_attack(
            policy_star=policy_star, mine_fn=lambda u, r, log: xu_stoller.mine(u, r, log),
            base_log=base_log, target=target, adversary=adversary,
        )
        dt = time.time() - t0
        violated = cert.certified and trace.success
        print(f"  {target}: certified={cert.certified} attack_success={trace.success} "
              f"poison_tried={len(trace.delta)} ({dt:.1f}s) -> "
              f"{'CERTIFICATE VIOLATED!!' if violated else 'holds'}")
        writer.add(ExperimentRecord(
            problem="P4_certificate", dataset=name, target=str(target), knowledge="white_box",
            beta=1.0, poison_size=len(trace.delta),
            poison_fraction=len(trace.delta) / len(base_log) if base_log else 0.0,
            success=trace.success, seconds=dt,
            extra={"analysis": "stress_test", "certified": cert.certified, "violated": violated,
                   "adversary_size": len(controlled)},
        ))


def utility_cost(writer: ResultsWriter, name: str, withhold_fraction: float = 0.5):
    path = os.path.join(DATASET_DIR, f"{name}.abac")
    policy_star = parse_abac_file(path)
    full_log = enumerate_full_log(policy_star)

    rng = random.Random(1)
    n_withhold = int(round(withhold_fraction * len(full_log.grants)))
    withhold = set(rng.sample(full_log.grants, k=n_withhold))
    sparse_log = [q for q in full_log.grants if q not in withhold]

    t0 = time.time()
    strict_policy = xu_stoller.mine(policy_star.users, policy_star.resources, sparse_log)
    strict_dt = time.time() - t0
    strict_report = compare_to_ground_truth(strict_policy, full_log)
    print(f"\n=== {name} utility cost ({withhold_fraction:.0%} clean-log sparsity) ===")
    print(f"  strict (certified): rules={strict_policy.num_rules()} over={strict_report.over_privilege_count} "
          f"under={strict_report.under_privilege_count} ({strict_dt:.1f}s)")
    writer.add(ExperimentRecord(
        problem="P4_certificate", dataset=name, target="N/A", knowledge="N/A", tau=1.0, seconds=strict_dt,
        rule_count_after=strict_policy.num_rules(), over_privilege_after=strict_report.over_privilege_count,
        extra={"analysis": "utility_cost", "miner": "strict_xu_stoller",
               "under_privilege": strict_report.under_privilege_count, "withhold_fraction": withhold_fraction},
    ))

    for tau in UTILITY_TAUS:
        t0 = time.time()
        m = rhapsody.mine(policy_star.users, policy_star.resources, sparse_log, min_reliability=tau)
        dt = time.time() - t0
        r = compare_to_ground_truth(m, full_log)
        print(f"  rhapsody tau={tau}: rules={m.num_rules()} over={r.over_privilege_count} "
              f"under={r.under_privilege_count} ({dt:.1f}s)")
        writer.add(ExperimentRecord(
            problem="P4_certificate", dataset=name, target="N/A", knowledge="N/A", tau=tau, seconds=dt,
            rule_count_after=m.num_rules(), over_privilege_after=r.over_privilege_count,
            extra={"analysis": "utility_cost", "miner": "rhapsody",
                   "under_privilege": r.under_privilege_count, "withhold_fraction": withhold_fraction},
        ))


def run():
    writer = ResultsWriter("p4_certificate", RESULTS_DIR)
    for name in DATASETS:
        stress_test(writer, name)
        utility_cost(writer, name)
    writer.flush()


if __name__ == "__main__":
    run()
