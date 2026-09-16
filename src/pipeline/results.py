"""Structured result records for all experiments, so results are aggregatable into
paper tables/figures instead of living only in console print output.

One flat schema covers every problem (P1 rule-generalization, P2 MLBAC, P3 retention,
P4 certificate) by leaving problem-specific fields in `extra` -- keeps the CSV usable
with a single reader while still capturing everything each experiment produces.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

CORE_FIELDS = [
    "problem", "dataset", "target", "knowledge", "tau", "beta",
    "poison_size", "poison_fraction", "success", "baseline_already_granted",
    "over_privilege_before", "over_privilege_after", "collateral_over_privilege",
    "rule_count_before", "rule_count_after", "seconds",
]


@dataclass
class ExperimentRecord:
    problem: str  # "P1_rule_generalization" | "P2_mlbac" | "P3_retention" | "P4_certificate"
    dataset: str
    target: str
    knowledge: str = "black_box"
    tau: Optional[float] = None
    beta: Optional[float] = None
    poison_size: int = 0
    poison_fraction: float = 0.0
    success: bool = False
    baseline_already_granted: Optional[bool] = None
    over_privilege_before: Optional[int] = None
    over_privilege_after: Optional[int] = None
    collateral_over_privilege: Optional[int] = None
    rule_count_before: Optional[int] = None
    rule_count_after: Optional[int] = None
    seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class ResultsWriter:
    """Accumulates ExperimentRecords and writes them to CSV + JSON on flush()."""

    def __init__(self, name: str, out_dir: str):
        self.name = name
        self.out_dir = out_dir
        self.records: List[ExperimentRecord] = []

    def add(self, record: ExperimentRecord) -> None:
        self.records.append(record)

    def flush(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        json_path = os.path.join(self.out_dir, f"{self.name}.json")
        csv_path = os.path.join(self.out_dir, f"{self.name}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2, default=str)

        extra_keys = sorted({k for r in self.records for k in r.extra})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CORE_FIELDS + extra_keys)
            for r in self.records:
                row = [getattr(r, f) for f in CORE_FIELDS]
                row += [r.extra.get(k, "") for k in extra_keys]
                writer.writerow(row)
        print(f"[{self.name}] wrote {len(self.records)} records -> {csv_path}")
