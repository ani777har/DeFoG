"""Rank-correlation of trajectory probes against the final metric: whether
early stopping is viable. x = trajectory fraction elapsed, y = |Spearman
rank correlation| between a probe at that point and the final objective.

Usage:
    python src/analysis/probe_correlation.py OUTPUT_DIR [--objective average_ratio_mean]
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

JOIN_KEYS = ["num_step", "distortor", "eta", "omega"]

# Columns that identify a row rather than measure something.
NON_FEATURES = set(
    JOIN_KEYS
    + ["step", "t_frac", "t_distorted", "batch_id", "trial_seq", "map_valid_criterion"]
)


def _round_keys(df):
    df = df.copy()
    for k in ("eta", "omega"):
        if k in df:
            df[k] = df[k].astype(float).round(6)
    if "num_step" in df:
        df["num_step"] = df["num_step"].astype(int)
    if "distortor" in df:
        df["distortor"] = df["distortor"].astype(str)
    return df


def load(output_dir, results_csv=None, objective="average_ratio_mean"):
    probes = pd.read_csv(os.path.join(output_dir, "trajectory_probes.csv"))

    if results_csv is None:
        candidates = [
            p
            for p in glob.glob(os.path.join(output_dir, "*.csv"))
            if not p.endswith("trajectory_probes.csv")
        ]
        if not candidates:
            raise FileNotFoundError(f"No results CSV found in {output_dir}")
        results_csv = max(candidates, key=os.path.getsize)
        print(f"Using results file: {results_csv}")

    results = pd.read_csv(results_csv)
    results = results.loc[:, ~results.columns.str.match(r"^Unnamed")]
    if objective not in results.columns:
        raise KeyError(
            f"Objective '{objective}' not in {results_csv}. "
            f"Available: {sorted(c for c in results.columns if c.endswith('_mean'))}"
        )

    probes, results = _round_keys(probes), _round_keys(results)

    # Average across generation batches: `step` restarts at 0 for each batch.
    feature_cols = [
        c
        for c in probes.columns
        if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(probes[c])
    ]
    per_step = (
        probes.groupby(JOIN_KEYS + ["step", "t_frac"], as_index=False)[feature_cols]
        .mean()
    )

    merged = per_step.merge(
        results[JOIN_KEYS + [objective]].drop_duplicates(JOIN_KEYS),
        on=JOIN_KEYS,
        how="inner",
    )
    n_cfg = merged[JOIN_KEYS].drop_duplicates().shape[0]
    if n_cfg == 0:
        raise RuntimeError(
            "No configurations matched between the probe and results files -- "
            "check that both come from the same search run."
        )
    print(f"Matched {n_cfg} configurations, {len(feature_cols)} probe features.")
    return merged, feature_cols, objective


def correlate(merged, feature_cols, objective, n_bins=10):
    """|Spearman(feature at t, final objective)| for each feature and each t."""
    merged = merged.copy()
    # Bin by trajectory fraction so configs with different num_step line up.
    merged["t_bin"] = np.clip(
        np.ceil(merged["t_frac"] * n_bins) / n_bins, 1.0 / n_bins, 1.0
    )

    rows = []
    for t_bin, g in merged.groupby("t_bin"):
        # one value per configuration in this bin
        g = g.groupby(JOIN_KEYS, as_index=False)[feature_cols + [objective]].mean()
        if len(g) < 5:
            continue
        rec = {"t_bin": t_bin, "n_configs": len(g)}
        for f in feature_cols:
            x, y = g[f].values, g[objective].values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5 or np.allclose(x[ok], x[ok][0]):
                rec[f] = np.nan
                continue
            rec[f] = spearmanr(x[ok], y[ok]).correlation
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_sources(corr, feature_cols):
    """Paired pred_E vs sampled-state comparison, per structural statistic."""
    rows = []
    for hard in sorted(c for c in feature_cols if c.startswith("hard_")):
        soft = hard[len("hard_") :]
        if soft not in feature_cols:
            continue

        def summary(col):
            a = corr[["t_bin", col]].dropna()
            if a.empty:
                return np.nan, "-"
            hit = a.loc[a[col].abs() >= 0.7, "t_bin"]
            return a[col].abs().max(), f"{hit.min():.1f}" if len(hit) else "-"

        s_max, s_t = summary(soft)
        h_max, h_t = summary(hard)
        rows.append(
            {
                "statistic": soft,
                "pred_max": s_max,
                "pred_t@0.7": s_t,
                "hard_max": h_max,
                "hard_t@0.7": h_t,
                "winner": "pred" if (s_max or 0) >= (h_max or 0) else "sampled",
            }
        )
    return pd.DataFrame(rows)


def plot(corr, feature_cols, objective, out_path, top_k=10):
    absmax = (
        corr[feature_cols].abs().max().sort_values(ascending=False).dropna()
    )
    # Keep both families visible: the comparison is the point.
    pred_f = [c for c in absmax.index if not c.startswith("hard_")]
    hard_f = [c for c in absmax.index if c.startswith("hard_")]
    n_pred = max(1, top_k // 2)
    top = pred_f[:n_pred] + hard_f[: top_k - n_pred]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(top):
        is_hard = f.startswith("hard_")
        ax.plot(
            corr["t_bin"],
            corr[f].abs(),
            marker="s" if is_hard else "o",
            ms=4,
            lw=1.8,
            ls="--" if is_hard else "-",
            color=cmap(i % 10),
            label=f"{f} (max {absmax[f]:.2f})",
        )
    ax.axhline(0.7, ls="--", c="green", lw=1, alpha=0.7)
    ax.axhline(0.4, ls="--", c="red", lw=1, alpha=0.7)
    ax.text(0.01, 0.71, "prunable", color="green", fontsize=8, va="bottom")
    ax.text(0.01, 0.41, "too weak", color="red", fontsize=8, va="bottom")

    ax.set_xlabel("fraction of sampling trajectory elapsed")
    ax.set_ylabel(f"|Spearman rank corr.| with final {objective}")
    ax.set_title(
        f"Can a cheap probe rank configurations early?  "
        f"({corr['n_configs'].max()} configs)\n"
        f"solid = denoiser prediction (pred_E),  dashed = sampled state (hard_)",
        fontsize=10,
    )
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    return top


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", help="search version dir with trajectory_probes.csv")
    ap.add_argument("--results-csv", default=None, help="defaults to largest CSV in dir")
    ap.add_argument("--objective", default="average_ratio_mean")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    merged, feature_cols, objective = load(
        args.output_dir, args.results_csv, args.objective
    )
    corr = correlate(merged, feature_cols, objective, n_bins=args.bins)
    corr_path = os.path.join(args.output_dir, "probe_rank_correlation.csv")
    corr.to_csv(corr_path, index=False)
    print(f"Wrote {corr_path}")

    top = plot(
        corr,
        feature_cols,
        objective,
        os.path.join(args.output_dir, "probe_rank_correlation.png"),
    )

    cmp_df = compare_sources(corr, feature_cols)
    if not cmp_df.empty:
        cmp_path = os.path.join(args.output_dir, "probe_source_comparison.csv")
        cmp_df.to_csv(cmp_path, index=False)
        print(f"\nDenoiser prediction vs sampled state (-> {cmp_path}):")
        print(cmp_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        wins = cmp_df["winner"].value_counts()
        print(
            f"  pred_E wins {int(wins.get('pred', 0))} of {len(cmp_df)} statistics"
        )

    print("\nEarliest trajectory fraction at which each top probe reaches |rho|:")
    print(f"{'probe':<24} {'>=0.4':>8} {'>=0.7':>8} {'max|rho|':>9}")
    for f in top:
        a = corr[["t_bin", f]].dropna()
        first = lambda thr: (  # noqa: E731
            f"{a.loc[a[f].abs() >= thr, 't_bin'].min():.1f}"
            if (a[f].abs() >= thr).any()
            else "-"
        )
        print(f"{f:<24} {first(0.4):>8} {first(0.7):>8} {a[f].abs().max():>9.2f}")


if __name__ == "__main__":
    main()
