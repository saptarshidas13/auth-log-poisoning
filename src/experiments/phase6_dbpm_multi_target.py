"""Step B: multi-target statistics for the SACMAT 2026 MDL noisy-log miner --
matching the treatment given to Rhapsody in pipeline_p1_multi_target_sweep.py.

Only run on workforce/edocument: the small ABAC-Lab datasets already collapse to a
degenerate blanket-deny policy for this miner regardless of any attack (see
project-status memory), so an exploitability rate there would be measuring a
different (non-)phenomenon. Sample sizes are much smaller than the Rhapsody sweep
because each DBPM mine() call costs ~40-130s here vs. sub-second for Rhapsody.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abac.logs import build_partial_log, enumerate_full_log
from abac.parser import parse_abac_file
from attack import Adversary, find_near_miss_targets, greedy_rule_generalization_attack
from miners import domain_based
from pipeline import ExperimentRecord, ResultsWriter

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# (num_targets, beta)
SWEEP_CONFIG = {
    "workforce": (5, 0.05),
    "edocument": (2, 0.02),
}


def run_dataset(writer: ResultsWriter, name: str):
    num_targets, beta = SWEEP_CONFIG[name]
    path = os.path.join(DATASET_DIR, f"{name}.abac")
    policy_star = parse_abac_file(path)
    full_log = enumerate_full_log(policy_star)
    actions = sorted(policy_star.actions())
    role_attr = "role" if name == "edocument" else "position"

    near_misses = find_near_miss_targets(policy_star, limit=200_000)
    import random
    rng = random.Random(7)
    sample = rng.sample(near_misses, k=min(num_targets, len(near_misses)))

    print(f"\n=== {name} === {len(near_misses)} total near-miss targets, sampling {len(sample)}, beta={beta}")
    poison_sizes = []
    n_success = n_baseline_granted = n_fail = 0

    for ti, nm in enumerate(sample):
        target = nm.target
        role = policy_star.users[target[0]].attrs.get(role_attr)
        controlled = {uid for uid, e in policy_star.users.items() if e.attrs.get(role_attr) == role}
        adversary = Adversary(controlled_uids=frozenset(controlled), knowledge="grey_box", beta=beta)
        base_log = build_partial_log(full_log, controlled, target[1], withhold_fraction=1.0, seed=0)

        def mine_fn(users, resources, log_grants, _fl=full_log, _actions=actions):
            return domain_based.mine(users, resources, log_grants, _fl.denials, _actions, seed=0)

        t0 = time.time()
        trace = greedy_rule_generalization_attack(
            policy_star=policy_star, mine_fn=mine_fn, base_log=base_log, target=target, adversary=adversary,
        )
        dt = time.time() - t0
        print(f"  target {ti + 1}/{len(sample)} {target} (adversary_size={len(controlled)}): "
              f"success={trace.success} baseline_granted={trace.baseline_already_granted} "
              f"poison_size={len(trace.delta)} calls={len(trace.history) - 1} ({dt:.1f}s)")

        if trace.baseline_already_granted:
            n_baseline_granted += 1
        elif trace.success:
            n_success += 1
            poison_sizes.append(len(trace.delta))
        else:
            n_fail += 1

        writer.add(ExperimentRecord(
            problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="grey_box",
            beta=beta, poison_size=len(trace.delta),
            poison_fraction=len(trace.delta) / len(base_log) if base_log else 0.0,
            success=trace.success, baseline_already_granted=trace.baseline_already_granted, seconds=dt,
            extra={"rq": "DBPM_multi_target", "miner": "MDL_AutoPart"},
        ))
        writer.flush()

    genuine_n = n_success + n_fail
    rate = n_success / genuine_n if genuine_n else float("nan")
    print(f"\n  {name} summary: exploit_rate={rate:.1%} ({n_success}/{genuine_n} genuine trials, "
          f"{n_baseline_granted} already-granted excluded)")
    if poison_sizes:
        print(f"  poison size: mean={statistics.mean(poison_sizes):.1f} "
              f"std={statistics.pstdev(poison_sizes) if len(poison_sizes) > 1 else 0:.1f} "
              f"median={statistics.median(poison_sizes):.0f}")


def run():
    writer = ResultsWriter("p1_dbpm_multi_target", RESULTS_DIR)
    for name in SWEEP_CONFIG:
        run_dataset(writer, name)
    writer.flush()


if __name__ == "__main__":
    run()
