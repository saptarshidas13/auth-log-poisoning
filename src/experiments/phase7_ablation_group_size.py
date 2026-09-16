"""TDSC gap item #1: a deliberate, controlled ablation over colluding-group size,
rather than group size being an incidental byproduct of whichever role happens to
exist in a dataset. For a fixed target and reliability threshold, sample RANDOM
subsets of increasing size from the eligible role-group and measure the minimal
poison size needed (via the existing exponential-probe + binary search), across
several random subset draws per size to get variance.
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abac.logs import build_partial_log, enumerate_full_log
from abac.parser import parse_abac_file
from attack import Adversary, find_near_miss_targets, greedy_rule_generalization_attack
from miners import rhapsody
from pipeline import ExperimentRecord, ResultsWriter

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def run(name: str = "university", tau: float = 0.5, group_sizes=(2, 3, 4, 5, 6, 7, 8, 9, 10), num_seeds: int = 10):
    path = os.path.join(DATASET_DIR, f"{name}.abac")
    policy_star = parse_abac_file(path)
    full_log = enumerate_full_log(policy_star)

    nm = find_near_miss_targets(policy_star, limit=1)[0]
    target = nm.target
    op_gain = target[1]
    role_attr = "role" if name == "edocument" else "position"
    role = policy_star.users[target[0]].attrs.get(role_attr)
    full_group = sorted({uid for uid, e in policy_star.users.items() if e.attrs.get(role_attr) == role})
    print(f"=== {name} === target={target} ceiling={nm.baseline_reliability:.3f} "
          f"full_role_group_size={len(full_group)} tau={tau}")

    writer = ResultsWriter("p7_ablation_group_size", RESULTS_DIR)
    resolved_sizes = [s if s is not None else len(full_group) for s in group_sizes]

    for size in resolved_sizes:
        size = min(size, len(full_group))
        poison_sizes, successes = [], []
        for seed in range(num_seeds):
            rng = random.Random(1000 * size + seed)
            group = rng.sample(full_group, k=size) if size < len(full_group) else list(full_group)
            # target's own uid must be controllable for a coherent escalation story
            if target[0] not in group:
                group[0] = target[0]
            controlled = frozenset(group)
            adversary = Adversary(controlled_uids=controlled, knowledge="grey_box", beta=0.95)
            base_log = build_partial_log(full_log, controlled, op_gain, withhold_fraction=1.0, seed=seed)

            trace = greedy_rule_generalization_attack(
                policy_star=policy_star,
                mine_fn=lambda u, r, log: rhapsody.mine(u, r, log, min_reliability=tau),
                base_log=base_log, target=target, adversary=adversary,
            )
            if trace.baseline_already_granted:
                continue
            successes.append(trace.success)
            if trace.success:
                poison_sizes.append(len(trace.delta))

            writer.add(ExperimentRecord(
                problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="grey_box",
                tau=tau, beta=0.95, poison_size=len(trace.delta), success=trace.success,
                baseline_already_granted=trace.baseline_already_granted,
                extra={"rq": "ablation_group_size", "group_size": size, "seed": seed},
            ))

        rate = sum(successes) / len(successes) if successes else float("nan")
        if poison_sizes:
            mean_p, std_p = statistics.mean(poison_sizes), (statistics.pstdev(poison_sizes) if len(poison_sizes) > 1 else 0.0)
            poison_str = f"{mean_p:.1f}+/-{std_p:.1f} (n={len(poison_sizes)})"
        else:
            poison_str = "n/a"
        print(f"  group_size={size:>3}: success_rate={rate:.0%} poison_size={poison_str}")

    writer.flush()


if __name__ == "__main__":
    args = sys.argv[1:]
    name = args[0] if args else "university"
    tau = float(args[1]) if len(args) > 1 else 0.5
    run(name, tau)
