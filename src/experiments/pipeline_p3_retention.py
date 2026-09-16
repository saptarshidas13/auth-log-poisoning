"""Structured Problem 3 (retention poisoning) sweep across all ABAC-Lab datasets and a
range of min_uses retention thresholds, writing results/p3_retention.{csv,json}.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abac.coverage import policy_grants
from abac.logs import enumerate_full_log
from abac.parser import parse_abac_file
from attack import Adversary, retention_attack
from pipeline import ExperimentRecord, ResultsWriter
from tightening import build_role_based_overprivileged_policy

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DATASETS = ["university", "healthcare", "project-management", "workforce", "edocument"]
MIN_USES_VALUES = [1, 2, 3, 5, 10]


def run():
    writer = ResultsWriter("p3_retention", RESULTS_DIR)
    for name in DATASETS:
        path = os.path.join(DATASET_DIR, f"{name}.abac")
        policy_star = parse_abac_file(path)
        true_grants = policy_grants(policy_star)
        full_log = enumerate_full_log(policy_star)

        p0 = build_role_based_overprivileged_policy(policy_star, role_attr="position")
        over_privilege = sorted(p0 - true_grants)
        if not over_privilege:
            print(f"{name}: no role-based over-privilege found, skipping")
            continue

        # A handful of distinct over-privileged permissions, spread across users.
        seen_users = set()
        picks = []
        for req in over_privilege:
            if req[0] not in seen_users:
                picks.append(req)
                seen_users.add(req[0])
            if len(picks) >= 5:
                break

        print(f"\n=== {name} === P* grants={len(true_grants)} P0 grants={len(p0)} "
              f"over-privilege={len(over_privilege)}, testing {len(picks)} targets")
        for target in picks:
            adversary = Adversary(controlled_uids=frozenset({target[0]}), knowledge="black_box", beta=1.0)
            for min_uses in MIN_USES_VALUES:
                result = retention_attack(p0, full_log.grants, target, adversary, min_uses=min_uses)
                print(f"  {target} min_uses={min_uses}: retained_before={result.retained_without_poison} "
                      f"retained_after={result.retained_with_poison} success={result.success}")
                writer.add(ExperimentRecord(
                    problem="P3_retention",
                    dataset=name,
                    target=str(target),
                    knowledge="black_box",
                    beta=1.0,
                    poison_size=result.poison_size,
                    poison_fraction=result.poison_size / len(full_log.grants),
                    success=result.success,
                    baseline_already_granted=result.retained_without_poison,
                    extra={"min_uses": min_uses, "p0_grants": len(p0), "over_privilege_count": len(over_privilege)},
                ))
    writer.flush()


if __name__ == "__main__":
    run()
