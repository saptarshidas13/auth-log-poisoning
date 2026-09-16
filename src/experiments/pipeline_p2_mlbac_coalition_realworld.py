"""Real-world MLBAC coalition experiment (Section 5.3 / 8.7 of the paper): a lone
adversary on amazon1 has too few of their own legitimate rows to move the classifier,
so a coalition of similarly-restrictive users pools its rows instead (Algorithm 3).

amazon1's base grant rate is 94.2%, so a uid chosen at random is usually already
granted almost everywhere and admits no denied target. This script instead selects the
`--coalition-size` controlled uids with the FEWEST legitimate (label=1) rows for
op_index 0, forming a coalition of genuinely restrictive users, then runs the same
greedy exponential-probe-then-binary-search poison-sizing procedure as Algorithm 1/3
(oversample factor fixed at 1; the search variable is how many top-ranked pooled rows
are used) for both the black-box and white-box ranking strategies.

Writes results/p2_mlbac_coalition_realworld.{csv,json}.

Note on exact reproduction: the coalition-size and resulting minimal-poison-size
numbers quoted in the paper (66 users, 168 pooled rows, 100 poisoned rows) came from an
interactive exploratory run whose exact coalition-selection and target-selection order
was not persisted (see the paper's Reproducibility Appendix, supplementary Section S7,
and this repo's README). This script implements the same methodology (Algorithm 3)
with a documented, deterministic selection rule (fewest-grants-first, tie-broken by
uid) and reports whatever minimal poison size that rule finds. On a verification run
it reproduced the qualitative finding exactly: the beneficiary's own single legitimate
row is not enough to flip the target alone (`black_box_lone_adversary` record,
success=False), while pooling the 66-user coalition supplies a better-aligned row from
a different member and succeeds. The exact minimal poison size differs from the
paper's quoted figures because it depends on which of many possible denied targets is
selected, which the paper's own multi-target sweep (RQ1, Table S1) already shows varies
widely; treat the poison-size column as illustrative of the mechanism, not as a
bit-exact replication of the quoted headline number.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from attack import Adversary, black_box_mlbac_attack, white_box_mlbac_attack
from mlbac import evaluate, load_dlbac, train_mlbac
from pipeline import ExperimentRecord, ResultsWriter

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets", "3_dlbac_synthetic", "DlbacAlpha", "dataset"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DATASET_NAME = "amazon1"
OP_INDEX = 0
EPOCHS = 10


def grant_counts_by_uid(train, op_index: int) -> dict:
    grant_mask = train.labels[:, op_index] == 1.0
    counts: dict = {}
    for uid in train.uids[grant_mask]:
        counts[int(uid)] = counts.get(int(uid), 0) + 1
    return counts


def pick_restrictive_coalition(train, op_index: int, coalition_size: int) -> frozenset:
    """The `coalition_size` uids with the fewest legitimate rows for op_index,
    tie-broken by uid for determinism."""
    counts = grant_counts_by_uid(train, op_index)
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    chosen = [uid for uid, _ in ordered[:coalition_size]]
    return frozenset(chosen)


def pick_target(train, controlled: frozenset, op_index: int, baseline_model):
    """Try each coalition member in turn as the beneficiary (fewest-grants-first,
    matching the coalition-selection order), and use the first one for whom some
    resource, already granted to someone outside the coalition but not to the
    beneficiary, is predicted denied by the baseline model. Not every restrictive
    user's attribute profile is itself denied access broadly (some generic roles are
    granted almost everywhere, per the paper's own finding), so the search must range
    over candidate beneficiaries, not stop at the single most row-restrictive one."""
    grant_mask = train.labels[:, op_index] == 1.0
    counts = grant_counts_by_uid(train, op_index)
    outside_rows = np.nonzero(~np.isin(train.uids, list(controlled)) & grant_mask)[0]
    seen_res, candidate_resources = set(), []
    for i in outside_rows:
        res = tuple(train.res_attrs[i].tolist())
        if res in seen_res:
            continue
        seen_res.add(res)
        candidate_resources.append(res)
    ra_all = np.array(candidate_resources)

    for beneficiary in sorted(controlled, key=lambda u: (counts.get(u, 0), u)):
        beneficiary_rows = np.nonzero((train.uids == beneficiary) & grant_mask)[0]
        beneficiary_attrs = train.user_attrs[beneficiary_rows[0]].tolist()
        beneficiary_res_set = set(map(tuple, train.res_attrs[beneficiary_rows].tolist()))
        mask = np.array([res not in beneficiary_res_set for res in candidate_resources])
        if not mask.any():
            continue
        ua = np.tile(np.array(beneficiary_attrs), (mask.sum(), 1))
        ra = ra_all[mask]
        with torch.no_grad():
            logits = baseline_model(torch.from_numpy(ua), torch.from_numpy(ra))
            pred = (torch.sigmoid(logits) >= 0.5).float().numpy()[:, 0]
        denied_idx = np.nonzero(pred == 0)[0]
        if len(denied_idx) > 0:
            target_res_attrs = ra[denied_idx[0]].tolist()
            return beneficiary, beneficiary_attrs, target_res_attrs
    return None, None, None


def search_minimal_poison(attack_fn, train, target_user_attrs, target_res_attrs, op_index, adversary, pool_size):
    """Exponential probe then binary search over k_neighbors (Algorithm 1/3's sizing
    procedure), oversample factor fixed at 1."""
    n, n_max = 1, pool_size
    last_trace = None
    while n <= n_max:
        t0 = time.time()
        trace = attack_fn(
            train=train, target_user_attrs=target_user_attrs, target_res_attrs=target_res_attrs,
            op_index=op_index, adversary=adversary, k_neighbors=n, oversample_factor=1,
            train_kwargs={"epochs": EPOCHS}, seed=0,
        )
        dt = time.time() - t0
        print(f"    n={n}: poison_size={trace.poison_size} success={trace.success} ({dt:.1f}s)")
        last_trace = trace
        if trace.success:
            break
        n *= 2
    if last_trace is None or not last_trace.success:
        return last_trace
    lo, hi = max(1, n // 2), n
    while lo < hi:
        mid = (lo + hi) // 2
        trace = attack_fn(
            train=train, target_user_attrs=target_user_attrs, target_res_attrs=target_res_attrs,
            op_index=op_index, adversary=adversary, k_neighbors=mid, oversample_factor=1,
            train_kwargs={"epochs": EPOCHS}, seed=0,
        )
        print(f"    [binary search] n={mid}: success={trace.success}")
        if trace.success:
            hi = mid
            last_trace = trace
        else:
            lo = mid + 1
    return last_trace


def run(coalition_size: int = 66):
    train, test = load_dlbac(DATASET_NAME, DATASET_ROOT)
    grant_mask = train.labels[:, OP_INDEX] == 1.0
    print(f"=== {DATASET_NAME} === train={len(train)} test={len(test)} base_grant_rate={grant_mask.mean():.4f}")

    controlled = pick_restrictive_coalition(train, OP_INDEX, coalition_size)
    counts = grant_counts_by_uid(train, OP_INDEX)
    pool_rows = np.isin(train.uids, list(controlled)) & grant_mask
    print(f"coalition size={len(controlled)} combined candidate rows={int(pool_rows.sum())} "
          f"per-user grant counts (min/max)={min(counts.get(u, 0) for u in controlled)}/"
          f"{max(counts.get(u, 0) for u in controlled)}")

    baseline_model = train_mlbac(train, epochs=EPOCHS, seed=0)
    baseline_test = evaluate(baseline_model, test)
    print(f"baseline test accuracy={baseline_test.overall_accuracy:.4f}")

    beneficiary, beneficiary_attrs, target_res_attrs = pick_target(train, controlled, OP_INDEX, baseline_model)
    if target_res_attrs is None:
        print("no denied target found for this coalition; try a different --coalition-size")
        return
    print(f"beneficiary uid={beneficiary} target_res_attrs={target_res_attrs}")

    adversary = Adversary(controlled_uids=controlled, knowledge="black_box", beta=0.05)  # type: ignore
    pool_size = int(pool_rows.sum())
    if pool_size == 0:
        print("coalition has zero candidate rows; try a larger --coalition-size")
        return

    writer = ResultsWriter("p2_mlbac_coalition_realworld", RESULTS_DIR)

    print("  --- lone-adversary baseline (beneficiary acting alone) ---")
    lone_adversary = Adversary(controlled_uids=frozenset({beneficiary}), knowledge="black_box", beta=0.4)  # type: ignore
    lone_pool_size = int((np.isin(train.uids, [beneficiary]) & grant_mask).sum())
    lone_trace = search_minimal_poison(
        black_box_mlbac_attack, train, beneficiary_attrs, target_res_attrs, OP_INDEX,
        lone_adversary, max(lone_pool_size, 1),
    ) if lone_pool_size > 0 else None
    print(f"    lone adversary: own_rows={lone_pool_size} "
          f"success={bool(lone_trace and lone_trace.success)}")
    writer.add(ExperimentRecord(
        problem="P2_mlbac_coalition", dataset=DATASET_NAME,
        target=f"uid={beneficiary},res={target_res_attrs},op={OP_INDEX}",
        knowledge="black_box_lone_adversary", beta=0.4,
        poison_size=lone_trace.poison_size if lone_trace else 0,
        poison_fraction=(lone_trace.poison_size / len(train)) if lone_trace else 0.0,
        success=bool(lone_trace and lone_trace.success),
        baseline_already_granted=bool(lone_trace and lone_trace.baseline_prediction == 1),
        extra={"coalition_size": 1, "pool_size": lone_pool_size},
    ))

    for label, attack_fn, knowledge in (
        ("black_box", black_box_mlbac_attack, "black_box"),
        ("white_box", white_box_mlbac_attack, "white_box"),
    ):
        print(f"  --- {label} search ---")
        trace = search_minimal_poison(
            attack_fn, train, beneficiary_attrs, target_res_attrs, OP_INDEX, adversary, pool_size,
        )
        record = ExperimentRecord(
            problem="P2_mlbac_coalition", dataset=DATASET_NAME,
            target=f"uid={beneficiary},res={target_res_attrs},op={OP_INDEX}",
            knowledge=knowledge, beta=0.05,
            poison_size=trace.poison_size if trace else 0,
            poison_fraction=(trace.poison_size / len(train)) if trace else 0.0,
            success=bool(trace and trace.success),
            baseline_already_granted=bool(trace and trace.baseline_prediction == 1),
            extra={"coalition_size": len(controlled), "pool_size": pool_size,
                   "baseline_test_accuracy": baseline_test.overall_accuracy},
        )
        if trace and trace.success:
            poisoned_model = train_mlbac(trace.poisoned_train, epochs=EPOCHS, seed=0)
            poisoned_test = evaluate(poisoned_model, test)
            record.extra["poisoned_test_accuracy"] = poisoned_test.overall_accuracy
            record.extra["test_accuracy_delta"] = poisoned_test.overall_accuracy - baseline_test.overall_accuracy
            print(f"    stealth: test accuracy {baseline_test.overall_accuracy:.4f} -> "
                  f"{poisoned_test.overall_accuracy:.4f}")
        writer.add(record)
    writer.flush()


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 66
    run(size)
