"""Aggregate checkpoints over seeds: curves, final summary with bootstrap CIs, 2x2 effects.

Statistical unit = the training seed (runs are independent; episodes and time steps
within a run are not). Intervals are 95% percentile bootstraps over seeds; with 5 seeds
per cell they are too narrow (anti-conservative), so simple effects also get an exact
two-sided permutation test (perm_p), which remains valid for tiny samples.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = [("ddpg", "no", "DDPG"), ("ddpg", "yes", "DDPG + LayerNorm"),
           ("td3", "no", "TD3"), ("td3", "yes", "TD3 + LayerNorm")]
LABELS = {(a, l): name for a, l, name in CONFIGS}
COLORS = {("ddpg", "no"): "#2563A4", ("ddpg", "yes"): "#7FB2E5",
          ("td3", "no"): "#C2410C", ("td3", "yes"): "#F59E6B"}
STYLES = {"no": "-", "yes": "--"}
FINAL_METRICS = ["mean_return", "q1_bias", "q1_rel_bias", "q1_mae", "q1_rmse",
                 "qmin_bias", "qmin_rel_bias", "timeout_episodes"]
EFFECT_METRICS = ["mean_return", "q1_bias", "q1_rel_bias", "q1_mae"]
N_BOOT = 10_000


def load_results(result_dir: Path) -> pd.DataFrame:
    paths = sorted(result_dir.glob("*/evaluations.csv"))
    if not paths:
        raise FileNotFoundError(f"No evaluations.csv in {result_dir}. Run train_all.py first.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["layer_norm"] = df["layer_norm"].astype(str)
    return df


def bootstrap_mean_ci(values, rng, n_boot=N_BOOT, level=0.95):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(v.mean()), float(lo), float(hi)


def final_by_seed(df: pd.DataFrame, last_k: int = 3) -> pd.DataFrame:
    """Per run: average of the last ``last_k`` checkpoints (less noisy than the last one)."""
    rows = []
    for (alg, ln, seed), part in df.groupby(["algorithm", "layer_norm", "seed"]):
        part = part[part.step > 0].sort_values("step").tail(last_k)
        if part.empty:
            continue
        row = {"algorithm": alg, "layer_norm": ln, "seed": seed,
               "first_step": int(part.step.min()), "last_step": int(part.step.max()),
               "n_checkpoints": len(part)}
        for m in FINAL_METRICS:
            if m in part:
                row[m] = float(part[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(final: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for alg, ln, label in CONFIGS:
        part = final[(final.algorithm == alg) & (final.layer_norm == ln)]
        row = {"config": label, "algorithm": alg, "layer_norm": ln, "n_seeds": len(part)}
        for m in FINAL_METRICS:
            if m in part:
                mean, lo, hi = bootstrap_mean_ci(part[m], rng)
                row[m] = mean
                row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_pvalue(a, b, max_exact=200_000, n_mc=50_000, seed=0) -> float:
    """Two-sided permutation test for a difference of means between two groups.

    Exact (all relabelings) when feasible, Monte-Carlo otherwise. With 5 vs 5 runs
    the smallest attainable p-value is 2/252 = 0.0079 (complete separation).
    Unlike the percentile bootstrap, it stays valid for very small samples.
    """
    from itertools import combinations
    from math import comb
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled, n = np.r_[a, b], len(a)
    obs = abs(a.mean() - b.mean()) - 1e-12
    if comb(len(pooled), n) <= max_exact:
        total = pooled.sum()
        hits = count = 0
        for idx in combinations(range(len(pooled)), n):
            sa = pooled[list(idx)].sum()
            diff = sa / n - (total - sa) / (len(pooled) - n)
            hits += abs(diff) >= obs
            count += 1
        return hits / count
    r = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_mc):
        perm = r.permutation(pooled)
        hits += abs(perm[:n].mean() - perm[n:].mean()) >= obs
    return (hits + 1) / (n_mc + 1)


SIMPLE_EFFECTS = {"TD3 - DDPG | sans LN": (("td3", "no"), ("ddpg", "no")),
                  "TD3 - DDPG | avec LN": (("td3", "yes"), ("ddpg", "yes")),
                  "LN - sans LN | DDPG": (("ddpg", "yes"), ("ddpg", "no")),
                  "LN - sans LN | TD3": (("td3", "yes"), ("td3", "no"))}


def factorial_effects(final: pd.DataFrame, rng, n_boot=N_BOOT) -> pd.DataFrame:
    """Main effects, simple effects and interaction of the 2x2 design, seeds resampled per cell."""
    cells = {}
    for alg, ln, _ in CONFIGS:
        part = final[(final.algorithm == alg) & (final.layer_norm == ln)]
        cells[(alg, ln)] = part
    columns = ["metric", "contrast", "estimate", "ci_lo", "ci_hi", "ci_excludes_0", "perm_p"]
    if any(c.empty for c in cells.values()):
        return pd.DataFrame(columns=columns)

    def contrasts(m):
        dn, dy, tn, ty = (m[("ddpg", "no")], m[("ddpg", "yes")],
                          m[("td3", "no")], m[("td3", "yes")])
        return {
            "TD3 - DDPG (moyenne sur LN)": (tn + ty - dn - dy) / 2,
            "LN - sans LN (moyenne sur algo)": (dy + ty - dn - tn) / 2,
            "Interaction algo x LN": (ty - tn) - (dy - dn),
            "TD3 - DDPG | sans LN": tn - dn,
            "TD3 - DDPG | avec LN": ty - dy,
            "LN - sans LN | DDPG": dy - dn,
            "LN - sans LN | TD3": ty - tn,
        }

    rows = []
    for metric in EFFECT_METRICS:
        values = {k: c[metric].dropna().to_numpy(float) for k, c in cells.items()}
        if any(len(v) == 0 for v in values.values()):
            continue
        point = contrasts({k: v.mean() for k, v in values.items()})
        boot = {k: rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
                for k, v in values.items()}
        boot_c = contrasts(boot)
        for name, est in point.items():
            b = boot_c[name]
            lo, hi = np.quantile(b, [0.025, 0.975])
            perm_p = float("nan")
            if name in SIMPLE_EFFECTS:
                x, y = SIMPLE_EFFECTS[name]
                perm_p = permutation_pvalue(values[x], values[y])
            rows.append({"metric": metric, "contrast": name, "estimate": float(est),
                         "ci_lo": float(lo), "ci_hi": float(hi),
                         "ci_excludes_0": bool(len(min(values.values(), key=len)) > 1
                                               and (lo > 0 or hi < 0)),
                         "perm_p": perm_p})
    return pd.DataFrame(rows, columns=columns)


def plot_curves(df: pd.DataFrame, output: Path, rng):
    metrics = [("mean_return", "Retour d'évaluation (non actualisé)", "return", True),
               ("q1_bias", "Biais signé  E[Q1 − G]", "bias", False),
               ("q1_rel_bias", "Biais relatif  E[Q1 − G] / E|G|", "rel_bias", False),
               ("q1_mae", "Erreur absolue moyenne |Q1 − G|", "mae", False)]
    for column, ylabel, fname, include_step0 in metrics:
        fig, ax = plt.subplots(figsize=(7.6, 2.95))
        for alg, ln, label in CONFIGS:
            part = df[(df.algorithm == alg) & (df.layer_norm == ln)]
            if not include_step0:
                part = part[part.step > 0]   # random critic at step 0: bias meaningless
            if part.empty:
                continue
            xs, ys, los, his = [], [], [], []
            for step, g in part.groupby("step"):
                m, lo, hi = bootstrap_mean_ci(g[column], rng, n_boot=2000)
                xs.append(step); ys.append(m); los.append(lo); his.append(hi)
            xs, ys, los, his = map(np.asarray, (xs, ys, los, his))
            color = COLORS[(alg, ln)]
            ax.plot(xs, ys, STYLES[ln], color=color, marker="o", markersize=3,
                    linewidth=1.6, label=label)
            if np.any(np.isfinite(los)):
                ax.fill_between(xs, los, his, color=color, alpha=0.13, linewidth=0)
        if fname in ("bias", "rel_bias"):
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set(xlabel="Interactions avec l'environnement", ylabel=ylabel)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7.5, ncol=4, frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
        fig.tight_layout()
        fig.savefig(output / f"{fname}.png", dpi=200)
        plt.close(fig)


def plot_all(df: pd.DataFrame, output: Path, last_k: int = 3):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    plot_curves(df, output, rng)
    final = final_by_seed(df, last_k)
    final.to_csv(output / "final_by_seed.csv", index=False)
    summary = summarize(final, rng)
    summary.to_csv(output / "final_summary.csv", index=False)
    effects = factorial_effects(final, rng)
    effects.to_csv(output / "effects.csv", index=False)
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--last-k", type=int, default=3,
                   help="number of final checkpoints averaged per run")
    a = p.parse_args()
    df = load_results(Path(a.results))
    final = plot_all(df, Path(a.out), a.last_k)
    print(f"Loaded {len(df)} checkpoints, {len(final)} runs")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(pd.read_csv(Path(a.out) / "final_summary.csv")[
            ["config", "n_seeds", "mean_return", "q1_bias", "q1_rel_bias", "q1_mae",
             "timeout_episodes"]].round(3).to_string(index=False))
        eff = pd.read_csv(Path(a.out) / "effects.csv")
        if not eff.empty:
            print(eff.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
