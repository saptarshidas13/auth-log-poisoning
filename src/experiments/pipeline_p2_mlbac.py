"""Structured Problem 2 sweep: black-box (metadata-proximity) vs white-box
(gradient-alignment) clean-label poisoning, across several DLBAC datasets, plus a
stealth (test-accuracy delta) check. Writes results/p2_mlbac.{csv,json}.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from attack import Adversary, black_box_mlbac_attack, hamming_distance, white_box_mlbac_attack
from mlbac import evaluate, load_dlbac, predict_one, train_mlbac
from pipeline import ExperimentRecord, ResultsWriter

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "3_dlbac_synthetic", "DlbacAlpha", "dataset"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DATASETS = ["u4k-r4k-auth11k", "u5k-r5k-auth12k", "u4k-r4k-auth21k"]
EPOCHS = 10


def pick_adversary_and_target(train, op_index):
    grant_mask = train.labels[:, op_index] == 1.0
    uid_counts = {}
    for uid in train.uids[grant_mask]:
        uid_counts[int(uid)] = uid_counts.get(int(uid), 0) + 1
    u_gain = max(uid_counts, key=uid_counts.get)
    u_gain_rows = np.nonzero((train.uids == u_gain) & grant_mask)[0]
    u_gain_attrs = train.user_attrs[u_gain_rows[0]].tolist()

    other_rows = np.nonzero((train.uids != u_gain) & grant_mask)[0]
    u_gain_res_set = set(map(tuple, train.res_attrs[u_gain_rows].tolist()))
    seen_res, candidates = set(), []
    for i in other_rows:
        res = tuple(train.res_attrs[i].tolist())
        if res in u_gain_res_set or res in seen_res:
            continue
        seen_res.add(res)
        d = hamming_distance(train.user_attrs[i], np.asarray(u_gain_attrs))
        candidates.append((d, i))
    candidates.sort(key=lambda x: x[0])
    return u_gain, u_gain_attrs, candidates


def run_dataset(writer: ResultsWriter, name: str, op_index: int = 0):
    train, test = load_dlbac(name, DATASET_ROOT)
    print(f"\n=== {name} (op_index={op_index}) === train={len(train)} test={len(test)}")

    u_gain, u_gain_attrs, candidates = pick_adversary_and_target(train, op_index)
    baseline_model = train_mlbac(train, epochs=EPOCHS, seed=0)
    baseline_test = evaluate(baseline_model, test)
    print(f"  adversary uid={u_gain}, baseline test accuracy={baseline_test.overall_accuracy:.4f}")

    target_res_attrs = None
    for d, i in candidates:
        res_attrs = train.res_attrs[i].tolist()
        if predict_one(baseline_model, u_gain_attrs, res_attrs)[op_index] == 0:
            target_res_attrs = res_attrs
            break
    if target_res_attrs is None:
        print("  every candidate resource already granted at baseline, skipping dataset")
        return

    adversary = Adversary(controlled_uids=frozenset({u_gain}), knowledge="black_box", beta=0.05)  # type: ignore

    for label, attack_fn, knowledge in (
        ("black_box", black_box_mlbac_attack, "black_box"),
        ("white_box", white_box_mlbac_attack, "white_box"),
    ):
        for k, factor in ((1, 5), (3, 5), (1, 20)):
            t0 = time.time()
            trace = attack_fn(
                train=train, target_user_attrs=u_gain_attrs, target_res_attrs=target_res_attrs,
                op_index=op_index, adversary=adversary, k_neighbors=k, oversample_factor=factor,
                train_kwargs={"epochs": EPOCHS}, seed=0,
            )
            dt = time.time() - t0
            print(f"  [{label}] k={k} oversample={factor}: poison_size={trace.poison_size} "
                  f"baseline_pred={trace.baseline_prediction} poisoned_pred={trace.poisoned_prediction} "
                  f"success={trace.success} ({dt:.1f}s)")

            record = ExperimentRecord(
                problem="P2_mlbac", dataset=name, target=f"uid={u_gain},res={target_res_attrs},op={op_index}",
                knowledge=knowledge, beta=0.05, poison_size=trace.poison_size,
                poison_fraction=trace.poison_size / len(train), success=trace.success,
                baseline_already_granted=(trace.baseline_prediction == 1), seconds=dt,
                extra={"k_neighbors": k, "oversample_factor": factor,
                       "baseline_test_accuracy": baseline_test.overall_accuracy},
            )
            if trace.success:
                poisoned_model = train_mlbac(trace.poisoned_train, epochs=EPOCHS, seed=0)
                poisoned_test = evaluate(poisoned_model, test)
                record.extra["poisoned_test_accuracy"] = poisoned_test.overall_accuracy
                record.extra["test_accuracy_delta"] = poisoned_test.overall_accuracy - baseline_test.overall_accuracy
                print(f"      stealth: test accuracy {baseline_test.overall_accuracy:.4f} -> "
                      f"{poisoned_test.overall_accuracy:.4f}")
            writer.add(record)


def run():
    writer = ResultsWriter("p2_mlbac", RESULTS_DIR)
    for name in DATASETS:
        run_dataset(writer, name)
    writer.flush()


if __name__ == "__main__":
    run()
