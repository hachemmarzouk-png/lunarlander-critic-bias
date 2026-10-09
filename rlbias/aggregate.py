"""Load per-run evaluation CSVs and compute seed-level statistics."""
from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_METRICS = [
    "mean_return", "q1_bias", "q1_rel_bias", "q1_mae", "q1_rmse",
    "qmin_bias", "qmin_rel_bias", "timeout_episodes",
]


def load_results(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*/evaluations.csv"))
    if not paths:
        raise FileNotFoundError(f"No evaluation CSVs under {directory}")
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    results["layer_norm"] = results["layer_norm"].astype(str)
    return results


def final_by_seed(data: pd.DataFrame, last_k: int = 3) -> pd.DataFrame:
    """Average the last evaluation checkpoints within each independent training run."""
    rows = []
    for (algorithm, ln, seed), group in data.groupby(["algorithm", "layer_norm", "seed"]):
        group = group[group.step > 0].sort_values("step").tail(last_k)
        if group.empty:
            continue
        row = {"algorithm": algorithm, "layer_norm": ln, "seed": seed,
               "first_step": int(group.step.min()), "last_step": int(group.step.max()),
               "n_checkpoints": len(group)}
        for metric in FINAL_METRICS:
            if metric in group:
                row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_pvalue(first, second) -> float:
    """Exact two-sided permutation p-value for the difference of independent means."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.r_[a, b]
    n = len(a)
    observed = abs(a.mean() - b.mean()) - 1e-12
    total = pooled.sum()
    hits = 0
    for choice in combinations(range(len(pooled)), n):
        sum_a = pooled[list(choice)].sum()
        difference = sum_a / n - (total - sum_a) / len(b)
        if abs(difference) >= observed:
            hits += 1
    return hits / comb(len(pooled), n)
