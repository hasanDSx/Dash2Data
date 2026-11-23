"""Orchestration: spec -> generated + calibrated + validated dataset."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .calibrator import calibrate, derive_column_targets
from .generator import generate_flat, plan_generation
from .measures import Evaluator, make_evaluator
from .schemas import DashboardSpec
from .validator import validate


@dataclass
class BuildResult:
    spec: DashboardSpec
    flat: pd.DataFrame
    n_rows: int
    seed: int
    attempts: int
    validation: pd.DataFrame
    summary: dict
    warnings: list = field(default_factory=list)


def build_dataset(spec: DashboardSpec, *, num_rows: int = 50, auto_rows: bool = True,
                  include_nulls: bool = False, seed: int = 42, max_attempts: int = 3) -> BuildResult:
    warnings: list = []
    col_targets, w = derive_column_targets(spec)
    warnings += w
    plan = plan_generation(spec, num_rows, auto_rows)
    warnings += plan.notes

    best = None
    for attempt in range(max_attempts):
        flat = generate_flat(spec, plan, col_targets, seed + attempt, include_nulls)
        ev: Evaluator = make_evaluator(flat, spec)
        cal_warn = calibrate(flat, spec, col_targets, ev)
        ev = make_evaluator(flat, spec)          # fresh evaluator (numeric columns changed)
        val, summ = validate(ev, spec)
        score = summ["mean_error_pct"]
        score = 1e9 if (isinstance(score, float) and math.isnan(score)) else score
        if best is None or score < best[0]:
            best = (score, flat, val, summ, cal_warn, attempt + 1)
        if summ["off"] == 0 and summ["mean_error_pct"] < 0.05:
            break
    _, flat, val, summ, cal_warn, attempts = best
    warnings += cal_warn
    return BuildResult(spec, flat, plan.n_rows, seed, attempts, val, summ, list(dict.fromkeys(warnings)))
