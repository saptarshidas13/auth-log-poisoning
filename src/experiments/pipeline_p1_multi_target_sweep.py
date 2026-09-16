"""Gap-closing experiment #1+#2: multi-target, multi-seed statistics for Problem 1.

Replaces "we found one exploitable near-miss target per dataset" with a measured
distribution: for a RANDOM sample of near-miss targets (not just the top-ranked ones,
to avoid selection bias), each tested under several base-log-withholding seeds and a
tau sweep, report:
  - exploitability rate per tau: what fraction of (target, seed) trials the attack
    genuinely flips (excluding trials where the target was already granted by benign
    sparsity alone -- see baseline_already_granted)
  - poison-size distribution (mean/median/std/min/max) among successful trials
  - a strict-Xu-Stoller control per target (run once, seed=0) as a large-N reinforcement
    of the Problem 4 certificate

This is what turns "attack succeeds" (existence claim) into "attack succeeds on X% of
near-miss targets at budget beta, needing on average Y +/- Z poison records" (a
measured, TDSC-caliber characterization) -- see [[project-poisoning-research-status]].
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
from miners import rhapsody, xu_stoller
from pipeline import ExperimentRecord, ResultsWriter

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
ROLE_ATTR_BY_DATASET = {
    "university": "position", "healthcare": "position", "project-management": "position",
    "workforce": "position", "edocument": "role",
}

# (num_targets, num_seeds, taus, beta) -- small datasets get a generous sweep since
# mining is sub-second there; workforce/edocument are bounded much more tightly since
# a single mining call can cost 5-60+ seconds at that scale.
SWEEP_CONFIG = {
    "university": (40, 3, (0.9, 0.75, 0.6, 0.5, 0.4, 0.3), 0.9),
    "healthcare": (40, 3, (0.9, 0.75, 0.6, 0.5, 0.4, 0.3), 0.9),
    "project-management": (40, 3, (0.9, 0.75, 0.6, 0.5, 0.4, 0.3), 0.9),
    "workforce": (5, 1, (0.7, 0.5), 0.1),
    "edocument": (3, 1, (0.7, 0.5), 0.05),
}


def run_dataset(writer: ResultsWriter, name: str):
    num_targets, num_seeds, taus, beta = SWEEP_CONFIG[name]
    path = os.path.join(DATASET_DIR, f"{name}.abac")
    policy_star = parse_abac_file(path)
    full_log = enumerate_full_log(policy_star)
    role_attr = ROLE_ATTR_BY_DATASET[name]

    all_near_misses = find_near_miss_targets(policy_star, limit=200_000)
    rng = random.Random(42)
    sample = rng.sample(all_near_misses, k=min(num_targets, len(all_near_misses)))
    print(f"\n=== {name} === {len(all_near_misses)} total near-miss targets, "
          f"sampling {len(sample)}, {num_seeds} seed(s), taus={taus}, beta={beta}")

    per_tau_trials = {tau: {"baseline_granted": 0, "success": 0, "fail": 0, "poison_sizes": []} for tau in taus}
    control_successes = 0
    t_start = time.time()

    verbose = len(sample) <= 10  # per-target timing for small samples (large/slow datasets)
    for ti, nm in enumerate(sample):
        t_target = time.time()
        target = nm.target
        role = policy_star.users[target[0]].attrs.get(role_attr)
        controlled = {uid for uid, e in policy_star.users.items() if e.attrs.get(role_attr) == role}
        adversary = Adversary(controlled_uids=frozenset(controlled), knowledge="grey_box", beta=beta)

        # Control: strict Xu-Stoller, once per target (seed=0) -- reinforces the
        # certificate at large N rather than the handful of targets tested before.
        base_log0 = build_partial_log(full_log, controlled, target[1], withhold_fraction=1.0, seed=0)
        control = greedy_rule_generalization_attack(
            policy_star=policy_star, mine_fn=lambda u, r, log: xu_stoller.mine(u, r, log),
            base_log=base_log0, target=target, adversary=adversary,
        )
        if control.success:
            control_successes += 1
            print(f"  !! CERTIFICATE VIOLATION on target {target} -- investigate immediately", flush=True)
        if verbose:
            print(f"  target {ti + 1}/{len(sample)} {target} (adversary_size={len(controlled)}): "
                  f"control done ({time.time() - t_target:.1f}s)", flush=True)

        for seed in range(num_seeds):
            base_log = build_partial_log(full_log, controlled, target[1], withhold_fraction=1.0, seed=seed)
            for tau in taus:
                t_trial = time.time()
                trace = greedy_rule_generalization_attack(
                    policy_star=policy_star,
                    mine_fn=lambda u, r, log, tau=tau: rhapsody.mine(u, r, log, min_reliability=tau),
                    base_log=base_log, target=target, adversary=adversary,
                )
                bucket = per_tau_trials[tau]
                if trace.baseline_already_granted:
                    bucket["baseline_granted"] += 1
                elif trace.success:
                    bucket["success"] += 1
                    bucket["poison_sizes"].append(len(trace.delta))
                else:
                    bucket["fail"] += 1
                if verbose:
                    print(f"    seed={seed} tau={tau}: success={trace.success} "
                          f"baseline_granted={trace.baseline_already_granted} poison_size={len(trace.delta)} "
                          f"({time.time() - t_trial:.1f}s)", flush=True)

                writer.add(ExperimentRecord(
                    problem="P1_rule_generalization", dataset=name, target=str(target), knowledge="grey_box",
                    tau=tau, beta=beta, poison_size=len(trace.delta),
                    poison_fraction=len(trace.delta) / len(base_log) if base_log else 0.0,
                    success=trace.success, baseline_already_granted=trace.baseline_already_granted,
                    extra={"rq": "RQ1_multi_target", "seed": seed, "ceiling_reliability": nm.baseline_reliability,
                           "target_index": ti},
                ))
        if not verbose and ((ti + 1) % 10 == 0 or ti == len(sample) - 1):
            print(f"  ... {ti + 1}/{len(sample)} targets done ({time.time() - t_start:.0f}s elapsed)", flush=True)
        writer.flush()  # incremental: never lose completed work if a later target hangs

    print(f"\n  control: {control_successes}/{len(sample)} targets flipped strict Xu-Stoller "
          f"(should be 0 per the Problem 4 certificate)")
    print(f"  {'tau':>6} | {'genuine N':>9} | {'exploit rate':>12} | {'poison size (mean+/-std, median)':>32}")
    print("  " + "-" * 70)
    for tau in taus:
        b = per_tau_trials[tau]
        genuine_n = b["success"] + b["fail"]
        rate = b["success"] / genuine_n if genuine_n else float("nan")
        if b["poison_sizes"]:
            mean_p = statistics.mean(b["poison_sizes"])
            std_p = statistics.pstdev(b["poison_sizes"]) if len(b["poison_sizes"]) > 1 else 0.0
            med_p = statistics.median(b["poison_sizes"])
            poison_str = f"{mean_p:.1f}+/-{std_p:.1f}, median={med_p:.0f} (n={len(b['poison_sizes'])})"
        else:
            poison_str = "n/a (no successes)"
        print(f"  {tau:>6} | {genuine_n:>9} | {rate:>11.1%} | {poison_str:>32} "
              f"[{b['baseline_granted']} already-granted excluded]")


def run():
    writer = ResultsWriter("p1_multi_target_sweep", RESULTS_DIR)
    for name in SWEEP_CONFIG:
        run_dataset(writer, name)
    writer.flush()


if __name__ == "__main__":
    run()
