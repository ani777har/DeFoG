"""Supplementary figures for the trajectory probes, on top of probe_report.py:

  9  validity over time (map_valid_frac has no panel of its own otherwise)
  10 |rho| heatmap of every probe at every t
  11 distance probes (diameter/aspl, soft vs hard)
  12 probe redundancy (which probes measure the same thing)

Run:  PYTHONPATH=. python analysis/probe_extra_figures.py <run_dir> [--out DIR]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.probe_report import (  # noqa: E402
    JOIN_KEYS,
    NON_FEATURES,
    _cfg_key,
    collect,
    label_top_bottom_n,
)

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
)
GOOD, BAD = C_BLUE, C_ORANGE
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
)
DIV_BR = LinearSegmentedColormap.from_list(
    "div_br", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a58a", "#d94f2b", "#7a2410"]
)


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _feature_cols(probes):
    return [
        c
        for c in probes.columns
        if c not in NON_FEATURES
        and pd.api.types.is_numeric_dtype(probes[c])
        and probes[c].notna().any()
    ]


def _per_cfg(probes):
    """Average duplicate trials/batches so each (config, step) is one row."""
    return probes.groupby(JOIN_KEYS + ["step", "t_frac"], as_index=False).mean(
        numeric_only=True
    )


# --------------------------------------------------------------- figure 9
def fig_validity(probes, labels, out):
    cols = [c for c in ("map_valid_frac", "map_connected_frac") if c in probes]
    if not cols:
        print("  skipped 9: no map_* columns (run with sample.probe_validity_every>0)")
        return
    g = _per_cfg(probes)
    fig, axes = plt.subplots(1, len(cols), figsize=(6.2 * len(cols), 4.6))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        for key, sub in g.groupby(JOIN_KEYS):
            lab = labels.get(_cfg_key(key))
            if lab is None:
                continue
            s = sub[["t_frac", col]].dropna()
            if s.empty:
                continue
            ax.plot(s.t_frac, s[col], lw=1.6, alpha=0.75,
                    color=GOOD if lab == "good" else BAD, zorder=2)
        for lab, color in (("good", GOOD), ("bad", BAD)):
            keys = {k for k, v in labels.items() if v == lab}
            sub = g[[_cfg_key(t) in keys
                     for t in g[JOIN_KEYS].itertuples(index=False, name=None)]]
            m = sub.groupby("t_frac")[col].mean().dropna()
            if not m.empty:
                ax.plot(m.index, m.values, lw=3.4, color=color, zorder=3,
                        path_effects=None, label=f"{lab} (mean of 10)")
        _style(ax, col, "fraction of sampling trajectory elapsed", "fraction of graphs")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(
        "Validity of the MAP graph during sampling — the only probe that is not a proxy",
        color=INK, fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "9_validity_over_time.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print("  wrote 9_validity_over_time.png")


# -------------------------------------------------------------- figure 10
def fig_rho_heatmap(probes, results, obj, out, top=34):
    g = _per_cfg(probes)
    feats = _feature_cols(g)
    res = results.set_index(JOIN_KEYS)[obj]

    ts = sorted(g.t_frac.unique())
    ts = [t for t in ts if t > 0]
    rows, index = [], []
    for f in feats:
        piv = g.pivot_table(index=JOIN_KEYS, columns="t_frac", values=f)
        piv = piv.reindex(res.index).dropna(how="all")
        if len(piv) < 5:
            continue
        vals = []
        for t in ts:
            if t not in piv.columns:
                vals.append(np.nan)
                continue
            pair = pd.concat([piv[t], res], axis=1).dropna()
            vals.append(
                abs(pair.corr(method="spearman").iloc[0, 1]) if len(pair) >= 5 else np.nan
            )
        if np.isfinite(vals).any():
            rows.append(vals)
            index.append(f)
    if not rows:
        print("  skipped 10: too few configs for rank correlation")
        return

    M = pd.DataFrame(rows, index=index, columns=[f"{t:.2f}" for t in ts])
    M = M.loc[M.max(axis=1).sort_values(ascending=False).index[:top]]

    fig, ax = plt.subplots(figsize=(1.05 + 0.42 * len(ts), 0.30 * len(M) + 2.0))
    im = ax.imshow(M.values, aspect="auto", cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, fontsize=8)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v) and v >= 0.7:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v >= 0.82 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("|Spearman rho| with final quality", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title(
        f"When does each probe become predictive?  (labelled cells reach |rho| >= 0.7)\n"
        f"objective: {obj}",
        color=INK, fontsize=11.5, pad=10,
    )
    ax.set_xlabel("fraction of sampling trajectory elapsed", color=INK_2, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "10_rho_heatmap.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote 10_rho_heatmap.png  ({len(M)} probes x {len(ts)} timepoints)")


# -------------------------------------------------------------- figure 11
def fig_distance(probes, labels, out):
    want = [("diameter_mean", "diameter (pred)"), ("hard_diameter_mean", "diameter (sampled)"),
            ("aspl_mean", "aspl (pred)"), ("hard_aspl_mean", "aspl (sampled)")]
    want = [(c, t) for c, t in want if c in probes.columns]
    if not want:
        print("  skipped 11: no diameter/aspl columns (log predates the distance probes)")
        return
    g = _per_cfg(probes)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, want):
        n_undef = 0
        for key, sub in g.groupby(JOIN_KEYS):
            lab = labels.get(_cfg_key(key))
            if lab is None:
                continue
            s = sub[["t_frac", col]]
            n_undef += int(s[col].isna().sum())
            s = s.dropna()  # undefined early steps left as a gap, not zeroed
            if s.empty:
                continue
            ax.plot(s.t_frac, s[col], lw=1.5, alpha=0.75,
                    color=GOOD if lab == "good" else BAD, zorder=2)
        for lab, color in (("good", GOOD), ("bad", BAD)):
            keys = {k for k, v in labels.items() if v == lab}
            sub = g[[_cfg_key(t) in keys
                     for t in g[JOIN_KEYS].itertuples(index=False, name=None)]]
            m = sub.groupby("t_frac")[col].mean().dropna()
            if not m.empty:
                ax.plot(m.index, m.values, lw=3.4, color=color, zorder=3,
                        label=f"{lab} (mean)")
        sub_t = title + (f"   ({n_undef} steps undefined — no reachable pair)"
                         if n_undef else "")
        _style(ax, sub_t, "fraction of trajectory elapsed", "hops")
        ax.legend(frameon=False, fontsize=9)

    for ax in axes[len(want):]:
        ax.set_visible(False)
    fig.suptitle(
        "Distance probes — shape, not size: a stringy graph and a compact one "
        "can share a density",
        color=INK, fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = os.path.join(out, "11_distance_probes.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print("  wrote 11_distance_probes.png")


# -------------------------------------------------------------- figure 12
def fig_redundancy(probes, out, t_at=0.5, top=30):
    g = _per_cfg(probes)
    ts = sorted(g.t_frac.unique())
    t = min(ts, key=lambda x: abs(x - t_at))
    snap = g[g.t_frac == t]
    feats = [c for c in _feature_cols(snap) if snap[c].nunique() > 1]
    if len(feats) < 3:
        print("  skipped 12: not enough varying probes")
        return
    var = snap[feats].std() / snap[feats].mean().abs().replace(0, np.nan)
    feats = list(var.dropna().sort_values(ascending=False).index[:top])
    C = snap[feats].corr(method="spearman")

    # Reorder by hierarchical clustering so redundant families form blocks.
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        D = (1.0 - C.fillna(0.0).values) / 2.0
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        order = dendrogram(
            linkage(squareform(D, checks=False), method="average"), no_plot=True
        )["leaves"]
        feats = [feats[i] for i in order]
        C = C.loc[feats, feats]
    except Exception as exc:
        print(f"  (12: clustering unavailable, leaving unordered: {exc})")

    fig, ax = plt.subplots(figsize=(0.40 * len(feats) + 3.2, 0.40 * len(feats) + 2.6))
    im = ax.imshow(C.values, cmap=DIV_BR, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Spearman rho between probes", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title(
        f"Which probes are measuring the same thing?  (across configs at t={t:.2f})\n"
        "deep blue / deep red blocks are redundant families — not independent evidence",
        color=INK, fontsize=11.5, pad=10,
    )
    fig.tight_layout()
    p = os.path.join(out, "12_probe_redundancy.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote 12_probe_redundancy.png  ({len(feats)} probes at t={t:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=10, help="configs per side for good/bad")
    args = ap.parse_args()

    probes, results, results_src = collect(args.paths)
    if probes is None or probes.empty:
        sys.exit("no trajectory_probes.csv found")
    if results_src:
        print(f"results CSV: {results_src}")
    out = args.out or os.path.join(
        args.paths[0] if os.path.isdir(args.paths[0]) else ".", "probe_report"
    )
    os.makedirs(out, exist_ok=True)

    labels, obj = {}, None
    if results is not None and not results.empty:
        vun = next((c for c in results.columns if "unic_non_iso_valid" in c), None)
        ratio = next((c for c in results.columns if c.startswith("average_ratio")), None)
        if ratio is None:
            ratio = next((c for c in results.columns if c.endswith("_ratio_mean")), None)
        obj = ratio
        if vun and ratio:
            labels, _quad = label_top_bottom_n(results, vun, ratio, n=args.n)

    print(f"probes: {len(probes)} rows, {probes[JOIN_KEYS].drop_duplicates().shape[0]} configs")
    print(f"labelled good/bad: {sum(v == 'good' for v in labels.values())}/"
          f"{sum(v == 'bad' for v in labels.values())}")
    print()

    if labels:
        fig_validity(probes, labels, out)
        fig_distance(probes, labels, out)
    else:
        print("  skipped 9 and 11: no good/bad labels (need a results CSV)")
    if obj and results is not None:
        fig_rho_heatmap(probes, results, obj, out)
    fig_redundancy(probes, out)
    print(f"\nAll figures in {out}")


if __name__ == "__main__":
    main()
