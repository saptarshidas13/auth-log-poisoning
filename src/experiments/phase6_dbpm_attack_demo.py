"""Test Problem 1's rule-generalization attack against the real SACMAT 2026 MDL
noisy-log miner (Zhang & Fong -- see src/miners/domain_based.py for the faithful
reimplementation and its documented adaptation/simplification notes).

Finding from validation (see conversation): this miner's model-complexity term
(k*M^2) is calibrated for large-entity-count datasets and collapses to a
near-blanket-deny policy (M=1-3, zero rules) on the small ABAC-Lab benchmarks
(university/healthcare/project-management, n~20-60). It only produces a
non-degenerate policy at workforce/edocument scale (n~600-800). This script
therefore runs the attack against workforce, where the baseline is meaningful.

Unlike Rhapsody, this miner is parameter-free (no reliability threshold to sweep)
-- the only lever available to the attack is Delta's size, so we run the greedy
search once per target rather than sweeping tau.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from abac.logs import build_partial_log, enumerate_full_log
from abac.parser import parse_abac_file
from attack import Adversary, find_near_miss_targets, greedy_rule_generalization_attack
from miners import domain_based

DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "1_abac_ground_truth", "ABAC-Lab", "DATASETS", "abac-datasets"
)


def make_dbpm_mine_fn(policy_star, actions, observed_denials):
    def mine_fn(users, resources, log_grants):
        return domain_based.mine(users, resources, log_grants, observed_denials, actions, seed=0)
    return mine_fn


def main(name: str = "workforce"):
    path = os.path.join(DATASET_DIR, f"{name}.abac")
    policy_star = parse_abac_file(path)
    full_log = enumerate_full_log(policy_star)
    actions = sorted(policy_star.actions())
    print(f"=== {name} === n={len(policy_star.users) + len(policy_star.resources)} k={len(actions)} "
          f"grants={len(full_log.grants)} denials={len(full_log.denials)}")

    nm = find_near_miss_targets(policy_star, limit=1)[0]
    target = nm.target
    print(f"target={target} (ceiling reliability against Rhapsody's mechanism: {nm.baseline_reliability:.3f} "
          f"-- not directly meaningful for this different miner family, just reusing the same target-discovery tool)")

    role = policy_star.users[target[0]].attrs.get("position")
    controlled = {uid for uid, e in policy_star.users.items() if e.attrs.get("position") == role}
    adversary = Adversary(controlled_uids=frozenset(controlled), knowledge="grey_box", beta=0.05)
    base_log = build_partial_log(full_log, controlled, target[1], withhold_fraction=1.0, seed=0)
    print(f"adversary_size={len(controlled)} base_log={len(base_log)} budget={int(0.05 * len(base_log))}")

    # Full closed-world denials (dataset is exactly enumerable) -- see the module
    # docstring in domain_based.py for why sparse random sampling under-detects
    # incompatibilities and was replaced with this.
    mine_fn = make_dbpm_mine_fn(policy_star, actions, full_log.denials)

    t0 = time.time()
    baseline = mine_fn(policy_star.users, policy_star.resources, base_log)
    print(f"baseline DBPM policy: domains={baseline.num_domains()} rules={baseline.num_rules()} "
          f"target_decision={baseline.decide(*target)} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    trace = greedy_rule_generalization_attack(
        policy_star=policy_star, mine_fn=mine_fn, base_log=base_log, target=target, adversary=adversary,
    )
    dt = time.time() - t0
    print(f"attack: success={trace.success} baseline_already_granted={trace.baseline_already_granted} "
          f"poison_size={len(trace.delta)} calls={len(trace.history) - 1} ({dt:.1f}s)")
    if trace.success:
        poisoned = mine_fn(policy_star.users, policy_star.resources, base_log + trace.delta)
        print(f"  poisoned DBPM policy: domains={poisoned.num_domains()} rules={poisoned.num_rules()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
