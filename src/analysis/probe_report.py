"""One-stop report over trajectory probe logs.

Point it at any number of probe CSVs (or directories containing them):

    python src/analysis/probe_report.py outputs/.../version_0
    python src/analysis/probe_report.py a.csv b.csv --out /tmp/report

  * 1+ configs -> trajectory plots, alive-vs-flat probes, convergence /
                  over-stochasticity check, soft-vs-hard smoothness
  * 5+ configs -> rank correlation against the final metric (early-pruning
                  viability)

The results CSV (final metrics) is auto-detected in the same directory when
present; correlations are skipped if it is not.
"""

import argparse
import glob
import os
import warnings

import matplotlib

warnings.filterwarnings("ignore")  # constant-input / all-NaN slices are expected here

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.probe_correlation import (
    JOIN_KEYS,
    NON_FEATURES,
    compare_sources,
    correlate,
    plot as plot_correlation,
)

# Probes worth a panel, in display order.
PANEL_PROBES = [
    "deg_var",
    "triangles",
    "transitivity",
    "clustering",
    "density",
    "lcc_frac",
    "n_components",
    "diameter",
    "aspl",
    "lambda_max",
    "disagree_frac",
    "entropy_e",
    "commit_e",
    "churn",
    "flips",
    "jaccard",
]


# --------------------------------------------------------------------- loading


def _is_probe_frame(df):
    return "step" in df.columns and "density_mean" in df.columns


def _is_results_frame(df):
    return "eta" in df.columns and "omega" in df.columns and "step" not in df.columns


def collect(paths):
    """Split the given files/dirs into probe frames and a results frame."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.csv")))
        else:
            files.append(p)

    probes, results, results_src = [], None, None
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        if _is_probe_frame(df):
            df["source_file"] = os.path.basename(f)
            probes.append(df)
        elif _is_results_frame(df) and (results is None or len(df) > len(results)):
            results, results_src = df, f

    if not probes:
        raise SystemExit(f"No probe CSVs found in {paths}")

    probes = pd.concat(probes, ignore_index=True)
    for k in ("eta", "omega"):
        probes[k] = probes[k].astype(float).round(6)
        if results is not None:
            results[k] = results[k].astype(float).round(6)

    # Collapse repeated trials of the same config into one averaged trajectory.
    n_dup = probes.groupby(JOIN_KEYS)["trial_seq"].nunique()
    n_repeated_configs = int((n_dup > 1).sum())
    if n_repeated_configs:
        print(
            f"[collect] {n_repeated_configs} configuration(s) were evaluated more "
            f"than once (e.g. a search-space boundary revisited) -- averaging "
            f"their repeats into one trajectory per configuration."
        )
        probes = probes.groupby(
            JOIN_KEYS + ["step"], as_index=False
        ).mean(numeric_only=True)

    results = dedup_results(results)

    return probes, results, results_src


def dedup_results(results):
    """Collapse repeated evaluations of the same config into one averaged row."""
    if results is None:
        return None
    n_dup_results = results.groupby(JOIN_KEYS).size()
    n_repeated_results = int((n_dup_results > 1).sum())
    if n_repeated_results:
        print(
            f"[dedup_results] {n_repeated_results} configuration(s) in the "
            f"results table were evaluated more than once -- averaging repeats "
            f"into one row per configuration."
        )
        results = results.groupby(JOIN_KEYS, as_index=False).mean(numeric_only=True)
    return results


def pick_objective(results, requested):
    if results is None:
        return None
    if requested and requested in results.columns:
        return requested
    for c in ("average_ratio_mean", "sampling/frac_unic_non_iso_valid_mean"):
        if c in results.columns:
            return c
    return None


def n_pairs_of(g):
    """Node-pair count, logged directly or recovered from density and degree."""
    if "n_pairs" in g and g["n_pairs"].notna().any():
        return float(g["n_pairs"].median())
    dens, deg = g["density_mean"], g["deg_mean_mean"]
    ok = (dens > 1e-9) & dens.notna() & deg.notna()
    if not ok.any():
        return np.nan
    n = float((deg[ok] / dens[ok]).median()) + 1.0
    return n * (n - 1) / 2.0


# -------------------------------------------------------------------- sections


def probe_diagnostics(probes, feature_cols):
    """Is each probe moving at all, and does it move monotonically?"""
    rows = []
    for f in feature_cols:
        rel_ranges, monos = [], []
        for _, g in probes.groupby(JOIN_KEYS):
            v = g.sort_values("step")[f].values
            v = v[np.isfinite(v)]
            if len(v) < 5:
                continue
            scale = max(abs(np.mean(v)), 1e-12)
            rel_ranges.append((v.max() - v.min()) / scale)
            monos.append(abs(spearmanr(np.arange(len(v)), v).correlation))
        if not rel_ranges:
            continue
        rr, mono = np.median(rel_ranges), np.nanmedian(monos)
        verdict = "flat" if rr < 0.05 else ("monotone" if mono > 0.7 else "noisy")
        rows.append(
            {"probe": f, "rel_range": rr, "monotonicity": mono, "verdict": verdict}
        )
    out = pd.DataFrame(rows)
    order = {"monotone": 0, "noisy": 1, "flat": 2}  # most useful first
    out["_o"] = out["verdict"].map(order)
    return out.sort_values(["_o", "rel_range"], ascending=[True, False]).drop(
        columns="_o"
    )


def settling_check(probes, results, objective):
    """Does the sampler settle, or is it still churning at the end?"""
    rows = []
    for key, g in probes.groupby(JOIN_KEYS):
        g = g.sort_values("step")
        npairs = n_pairs_of(g)
        rec = dict(zip(JOIN_KEYS, key))
        if "disagree_frac_mean" in g:
            d = g["disagree_frac_mean"].values
            rec["disagree_start"] = d[0]
            rec["disagree_end"] = d[-1]
        if "flips_mean" in g and np.isfinite(npairs):
            rec["final_flip_pct"] = 100.0 * g["flips_mean"].values[-1] / npairs
        if "churn_mean" in g:
            c = g["churn_mean"].values[1:]
            rec["churn_trend"] = (
                spearmanr(np.arange(len(c)), c).correlation if len(c) > 3 else np.nan
            )
        if "hard_jaccard_mean" in g:
            j = g["hard_jaccard_mean"].values
            rec["final_jaccard"] = j[-1]
            locked = np.flatnonzero(np.isfinite(j) & (j >= 0.95))
            rec["lock_frac"] = locked[0] / len(j) if len(locked) else np.nan
        fp = rec.get("final_flip_pct", np.nan)
        rec["settles"] = (
            "-" if not np.isfinite(fp) else ("yes" if fp < 1.0 else "STILL CHURNING")
        )
        if results is not None and objective:
            m = results
            for k, v in zip(JOIN_KEYS, key):
                if k in m.columns:
                    m = m[m[k] == v] if k != "distortor" else m[m[k].astype(str) == str(v)]
            rec[objective] = float(m[objective].iloc[0]) if len(m) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def soft_vs_hard_smoothness(probes):
    """Step-to-step bounciness, prediction vs sampled state: jitter = raw
    std(diff); roughness = jitter / std(value), scale-free."""
    rows = []
    for base in [
        c[len("hard_") : -len("_mean")]
        for c in probes.columns
        if c.startswith("hard_") and c.endswith("_mean")
    ]:
        soft_c, hard_c = f"{base}_mean", f"hard_{base}_mean"
        if soft_c not in probes.columns:
            continue

        def stats(col):
            jit, rgh = [], []
            for _, g in probes.groupby(JOIN_KEYS):
                v = g.sort_values("step")[col].values
                v = v[np.isfinite(v)]
                if len(v) < 5:
                    continue
                j = np.std(np.diff(v))
                jit.append(j)
                if np.std(v) > 1e-12:
                    rgh.append(j / np.std(v))
            return (
                np.median(jit) if jit else np.nan,
                np.median(rgh) if rgh else np.nan,
            )

        js, rs = stats(soft_c)
        jh, rh = stats(hard_c)
        rows.append(
            {
                "statistic": base,
                "jitter_pred": js,
                "jitter_sampled": jh,
                "jitter_ratio": jh / js if js else np.nan,
                "roughness_pred": rs,
                "roughness_sampled": rh,
                "smoother": "pred" if (rs or np.inf) <= (rh or np.inf) else "sampled",
            }
        )
    return pd.DataFrame(rows)


def label_top_bottom_n(results, vun_col, ratio_col, n=10):
    """Label the N best and N worst configs (by combined VUN+ratio rank);
    everything in between is excluded."""
    have_vun = vun_col in results.columns
    have_ratio = ratio_col in results.columns
    if not (have_vun or have_ratio):
        return {}, {"good": 0, "bad": 0, "excluded": 0}

    df = results.reset_index(drop=True).copy()
    parts = []
    if have_ratio:
        parts.append(df[ratio_col].astype(float).rank(pct=True))       # high ratio = bad
    if have_vun:
        parts.append(1.0 - df[vun_col].astype(float).rank(pct=True))   # low vun = bad
    badness = sum(parts) / len(parts)
    order = badness.sort_values()

    n = min(n, len(order) // 2)
    best_idx, worst_idx = order.index[:n], order.index[-n:]
    label = {}
    for idx in best_idx:
        label[_cfg_key(df.loc[idx])] = "good"
    for idx in worst_idx:
        label[_cfg_key(df.loc[idx])] = "bad"
    quad = {
        "good": len(best_idx),
        "bad": len(worst_idx),
        "excluded": len(df) - len(best_idx) - len(worst_idx),
    }
    return label, quad


def good_bad_effect_table(probes, feature_cols, label, fracs=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Cohen's d between the good and bad groups, at each trajectory fraction."""
    rows = []
    for f in feature_cols:
        rec = {"probe": f}
        best_abs_d, best_t = 0.0, None
        for fr in fracs:
            g_good, g_bad = [], []
            for key, g in probes.groupby(JOIN_KEYS):
                lab = label.get(_cfg_key(key))
                if lab is None:
                    continue
                g = g.sort_values("step")
                idx = int(round(fr * (len(g) - 1)))
                v = g[f].values[idx]
                if not np.isfinite(v):
                    continue
                (g_good if lab == "good" else g_bad).append(v)
            if len(g_good) < 2 or len(g_bad) < 2:
                continue
            mg, mb = np.mean(g_good), np.mean(g_bad)
            pooled = np.sqrt((np.var(g_good, ddof=1) + np.var(g_bad, ddof=1)) / 2)
            d = (mb - mg) / pooled if pooled > 1e-12 else np.nan
            rec[f"mean_good_t{fr}"] = mg
            rec[f"mean_bad_t{fr}"] = mb
            rec[f"cohen_d_t{fr}"] = d
            if np.isfinite(d) and abs(d) > abs(best_abs_d):
                best_abs_d, best_t = d, fr
        rec["best_t"] = best_t
        rec["best_cohen_d"] = best_abs_d
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.reindex(out["best_cohen_d"].abs().sort_values(ascending=False).index)


def plot_good_vs_bad(probes, label, quad, out_path):
    """Good vs bad trajectories: thin individual lines, bold group means ± std."""
    avail = [p for p in PANEL_PROBES if f"{p}_mean" in probes.columns]
    ncol = 3
    nrow = int(np.ceil(len(avail) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    colors = {"good": "seagreen", "bad": "firebrick"}

    groups = {"good": [], "bad": []}
    for key, g in probes.groupby(JOIN_KEYS):
        lab = label.get(_cfg_key(key))
        if lab is not None:
            groups[lab].append(g.sort_values("step"))

    for i, probe in enumerate(avail):
        ax = axes[i // ncol][i % ncol]
        col = f"{probe}_mean"
        for lab in ("good", "bad"):
            traj = groups[lab]
            if not traj:
                continue
            for g in traj:
                ax.plot(g["step"], g[col], color=colors[lab], lw=0.8, alpha=0.25)
            n_steps = min(len(g) for g in traj)
            stack = np.stack([g[col].values[:n_steps] for g in traj])
            steps = traj[0]["step"].values[:n_steps]
            mean, std = stack.mean(0), stack.std(0)
            ax.plot(steps, mean, color=colors[lab], lw=2.6,
                     label=f"{lab} (n={len(traj)})")
            ax.fill_between(steps, mean - std, mean + std, color=colors[lab], alpha=0.12)
        ax.set_title(probe, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("step", fontsize=8)
        if i == 0:
            ax.legend(fontsize=8)
    for k in range(len(avail), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    subtitle = f"good={quad['good']}  bad={quad['bad']}  (excluded, in between: {quad['excluded']})"
    fig.suptitle(
        "Good vs bad trajectories  (thin = individual config, bold = group mean +/- std)\n"
        + subtitle,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    return out_path


# --------------------------------------------------------------------- figures


def _quality_map(results, objective, minimize):
    """Per-config objective value plus a colour scale that reads 'good = blue'."""
    if results is None or objective is None:
        return {}, None, None
    q = {}
    for _, r in results.iterrows():
        key = tuple(
            int(r["num_step"]) if k == "num_step" else
            (str(r[k]) if k == "distortor" else float(r[k]))
            for k in JOIN_KEYS
        )
        q[key] = float(r[objective])
    vals = np.array(list(q.values()))
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {}, None, None
    # heavy right tail on ratio-style objectives -> compress with log
    use_log = minimize and vals.min() > 0 and vals.max() / max(vals.min(), 1e-9) > 20
    tf = (lambda v: np.log10(v)) if use_log else (lambda v: v)
    lo, hi = tf(vals.min()), tf(vals.max())
    cmap = plt.get_cmap("viridis" if minimize else "viridis_r")

    def colour(key):
        v = q.get(key)
        if v is None or not np.isfinite(v):
            return (0.6, 0.6, 0.6, 1.0)
        x = (tf(v) - lo) / max(hi - lo, 1e-12)
        return cmap(x)

    label = f"{'log10 ' if use_log else ''}{objective}"
    return q, colour, (cmap, lo, hi, label, use_log)


def _cfg_key(row_or_key):
    if isinstance(row_or_key, tuple):
        k = row_or_key
    else:
        k = tuple(row_or_key[c] for c in JOIN_KEYS)
    return (int(k[0]), str(k[1]), float(k[2]), float(k[3]))


def _colorbar(fig, scale, ax_list):
    if scale is None:
        return
    cmap, lo, hi, label, _ = scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_list, shrink=0.85, pad=0.02)
    cb.set_label(label + "\n(dark = better)", fontsize=8)


def plot_landscape(probes, results, objective, minimize, out_path):
    """Where do the good configurations live in (eta, omega, distortor)?"""
    if results is None or objective is None:
        return None
    q, colour, scale = _quality_map(results, objective, minimize)
    if not q:
        return None
    dists = sorted({k[1] for k in q})
    ncol = min(3, len(dists))
    nrow = int(np.ceil(len(dists) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.1 * ncol + 1.3, 3.9 * nrow), squeeze=False,
        sharey=True, sharex=True, constrained_layout=True,
    )
    best = min(q, key=q.get) if minimize else max(q, key=q.get)
    for i, dname in enumerate(dists):
        ax = axes[i // ncol][i % ncol]
        keys = [k for k in q if k[1] == dname]
        ax.scatter(
            [k[2] for k in keys], [k[3] for k in keys],
            c=[colour(k) for k in keys], s=90, edgecolor="k", linewidth=0.4,
        )
        if best[1] == dname:
            ax.scatter([best[2]], [best[3]], s=320, facecolor="none",
                       edgecolor="red", linewidth=2, zorder=5)
            ax.annotate("best", (best[2], best[3]), textcoords="offset points",
                        xytext=(8, 8), color="red", fontsize=9)
        ax.set_title(f"{dname}  (n={len(keys)})", fontsize=10)
        ax.set_xlabel("eta")
        ax.grid(alpha=0.3)
    for k in range(len(dists), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for r in range(nrow):
        axes[r][0].set_ylabel("omega")
    fig.suptitle("Search landscape: which region of (eta, omega, schedule) works",
                 fontsize=12)
    _colorbar(fig, scale, axes.ravel().tolist())
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_settling(probes, results, objective, minimize, out_path):
    """Churn and flip rate over the trajectory, coloured by final quality."""
    q, colour, scale = _quality_map(results, objective, minimize)
    panels = [("churn_mean", "churn (prediction movement)"),
              ("flips_pct", "flips (% of node pairs, sampled state)"),
              ("hard_jaccard_mean", "edge stability (Jaccard vs previous step)"),
              ("disagree_frac_mean", "disagree_frac (model vs state)")]
    panels = [p for p in panels if p[0] == "flips_pct" or p[0] in probes.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels) + 1.4, 4.4),
                             squeeze=False, constrained_layout=True)
    for i, (col, title) in enumerate(panels):
        ax = axes[0][i]
        for key, g in probes.groupby(JOIN_KEYS):
            g = g.sort_values("step")
            y = (100.0 * g["flips_mean"] / n_pairs_of(g)) if col == "flips_pct" else g[col]
            ax.plot(g["step"], y, lw=1.4, alpha=0.85,
                    color=colour(_cfg_key(key)) if colour else "steelblue")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if col == "flips_pct":
            ax.axhline(1.0, ls="--", c="red", lw=1)
            ax.text(0.5, 1.05, "1% = settled", color="red", fontsize=8)
        if col == "hard_jaccard_mean":
            ax.axhline(1.0, ls="--", c="red", lw=1)
            ax.text(0.5, 1.0, "1.0 = edge set frozen", color="red", fontsize=8)
    fig.suptitle("Does the sampler settle?  Curves that stay high never converge",
                 fontsize=12)
    _colorbar(fig, scale, list(axes[0]))
    fig.savefig(out_path, dpi=150)
    return out_path


def rank_probes(corr, feature_cols, k=3):
    """Best-correlated probes, dropping near-duplicate profiles."""
    if corr is None or corr.empty:
        return []
    cand = [c for c in feature_cols if not c.startswith("hard_") and c in corr.columns]
    cand = sorted(cand, key=lambda c: -np.nan_to_num(corr[c].abs().max()))
    chosen = []
    for c in cand:
        prof = corr[c].abs().values
        if any(
            np.nanmax(np.abs(prof - corr[o].abs().values)) < 1e-6 for o in chosen
        ):
            continue
        chosen.append(c)
        if len(chosen) == k:
            break
    return chosen


def plot_probe_vs_quality(probes, results, objective, minimize, top_probes, out_path,
                          fracs=(0.25, 0.5, 0.75, 1.0)):
    """The money plot: probe value read at step t vs the configuration's final score."""
    q, colour, scale = _quality_map(results, objective, minimize)
    if not q or not top_probes:
        return None
    probes = probes.copy()
    rows = top_probes[:3]
    fig, axes = plt.subplots(len(rows), len(fracs),
                             figsize=(3.2 * len(fracs), 2.9 * len(rows)), squeeze=False)
    for r, probe in enumerate(rows):
        for c, fr in enumerate(fracs):
            ax = axes[r][c]
            xs, ys = [], []
            for key, g in probes.groupby(JOIN_KEYS):
                g = g.sort_values("step")
                idx = int(round(fr * (len(g) - 1)))
                v = g[probe].values[idx]
                score = q.get(_cfg_key(key))
                if score is not None and np.isfinite(v) and np.isfinite(score):
                    xs.append(v)
                    ys.append(score)
            if len(xs) >= 3:
                ax.scatter(xs, ys, s=42, c="steelblue", edgecolor="k", linewidth=0.3)
                rho = spearmanr(xs, ys).correlation
                ax.set_title(f"t={fr:.2f}   rho={rho:+.2f}", fontsize=9,
                             color=("darkgreen" if abs(rho) >= 0.7 else
                                    "darkorange" if abs(rho) >= 0.4 else "grey"))
            if minimize and len(ys) and min(ys) > 0 and max(ys) / min(ys) > 20:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(probe, fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("probe value", fontsize=8)
    fig.suptitle(
        f"Probe reading at step t  vs  final {objective}\n"
        "green title = strong enough to prune on, grey = not",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_soft_vs_hard(probes, sh, out_path):
    """Bar chart of the pred-vs-sampled jitter gap, plus one worked example."""
    if sh.empty:
        return None
    sh = sh.dropna(subset=["jitter_ratio"]).sort_values("jitter_ratio")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.barh(sh["statistic"], sh["jitter_ratio"], color="steelblue")
    ax.axvline(1.0, ls="--", c="k", lw=1)
    ax.set_xlabel("sampled-state jitter / prediction jitter")
    ax.set_title("How much noisier is the sampled state?", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    for i, (s, v) in enumerate(zip(sh["statistic"], sh["jitter_ratio"])):
        ax.text(v, i, f" {v:.1f}x", va="center", fontsize=8)

    base = sh["statistic"].iloc[-1]
    ax = axes[1]
    for j, (key, g) in enumerate(list(probes.groupby(JOIN_KEYS))[:4]):
        g = g.sort_values("step")
        col = plt.get_cmap("tab10")(j)
        ax.plot(g["step"], g[f"{base}_mean"], color=col, lw=1.8)
        ax.plot(g["step"], g[f"hard_{base}_mean"], color=col, lw=1.1, ls="--", alpha=0.7)
    ax.set_title(f"example: {base}  (solid=prediction, dashed=sampled)", fontsize=10)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    fig.suptitle("Why we read the denoiser's prediction, not the sampled state",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_diagnostics(diag, out_path):
    """Which probes move, and do they move monotonically."""
    d = diag[diag["probe"].str.endswith("_mean")].dropna(subset=["rel_range"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cols = {"monotone": "seagreen", "noisy": "darkorange", "flat": "lightgrey"}
    for verdict, g in d.groupby("verdict"):
        ax.scatter(np.maximum(g["rel_range"], 1e-3), g["monotonicity"].fillna(0),
                   s=70, label=verdict, color=cols.get(verdict, "grey"),
                   edgecolor="k", linewidth=0.4)
    for _, r in d.iterrows():
        ax.annotate(r["probe"].replace("_mean", ""),
                    (max(r["rel_range"], 1e-3), r["monotonicity"] if
                     np.isfinite(r["monotonicity"]) else 0),
                    fontsize=7, textcoords="offset points", xytext=(4, 3))
    ax.set_xscale("log")
    ax.axvline(0.05, ls="--", c="red", lw=1)
    ax.axhline(0.7, ls="--", c="green", lw=1)
    ax.set_xlabel("rel_range  (how much it moves; left of red line = flat)")
    ax.set_ylabel("monotonicity  (above green = clean trend)")
    ax.set_title("Probe screening: useful probes are top-right", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_trajectories(probes, results, objective, minimize, out_path, max_legend=6):
    """All probes over the trajectory, one line per config. Past max_legend
    configs, colour encodes final quality and a colorbar replaces the legend."""
    q, colour, scale = _quality_map(results, objective, minimize)
    configs = list(probes.groupby(JOIN_KEYS).groups.keys())
    avail = [p for p in PANEL_PROBES if f"{p}_mean" in probes.columns]
    few = len(configs) <= max_legend or colour is None
    ncol = 3
    nrow = int(np.ceil(len(avail) / ncol))

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.6 * ncol + (1.3 if not few else 0), 3.4 * nrow),
        squeeze=False, constrained_layout=not few,
    )
    cmap = plt.get_cmap("tab10")

    for i, probe in enumerate(avail):
        ax = axes[i // ncol][i % ncol]
        for j, key in enumerate(configs):
            g = probes
            for k, v in zip(JOIN_KEYS, key):
                g = g[g[k] == v]
            g = g.sort_values("step")
            col = cmap(j % 10) if few else colour(_cfg_key(key))
            lw = 1.7 if few else 1.0
            ax.plot(g["step"], g[f"{probe}_mean"], color=col, lw=lw, alpha=0.9)
            hard = f"hard_{probe}_mean"
            if hard in g.columns:
                ax.plot(g["step"], g[hard], color=col, lw=lw * 0.7, ls="--", alpha=0.5)
        ax.set_title(probe, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("step", fontsize=8)
    for k in range(len(avail), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    title = "Probe trajectories (solid = denoiser prediction, dashed = sampled state)"
    if few:
        handles = []
        for j, key in enumerate(configs):
            lab = dict(zip(JOIN_KEYS, key))
            txt = f"eta={lab['eta']:.1f} om={lab['omega']:.3f} {lab['distortor']}"
            if q:
                v = q.get(_cfg_key(key))
                if v is not None:
                    txt += f"  [{objective.split('/')[-1]}={v:.3g}]"
            handles.append(plt.Line2D([], [], color=cmap(j % 10), lw=2, label=txt))
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
                   frameon=False)
        fig.suptitle(title, fontsize=12)
        # cap the legend band so the panels can never be squeezed to nothing
        bottom = min(0.30, 0.04 + 0.045 * np.ceil(len(handles) / 2))
        fig.tight_layout(rect=[0, bottom, 1, 0.96])
    else:
        _colorbar(fig, scale, axes.ravel().tolist())
        fig.suptitle(f"{title}\n{len(configs)} configurations", fontsize=12)
    fig.savefig(out_path, dpi=150)
    return out_path


# ------------------------------------------------------------------------ main


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="probe CSVs and/or directories")
    ap.add_argument("--results", default=None, help="results CSV (else auto-detected)")
    ap.add_argument("--objective", default=None)
    ap.add_argument("--out", default=None, help="output dir (default: first path's dir)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument(
        "--all-features", action="store_true", help="include _std columns in section 2"
    )
    ap.add_argument("--minimize", action="store_true", help="lower objective is better")
    ap.add_argument("--maximize", action="store_true", help="higher objective is better")
    ap.add_argument("--vun-col", default="sampling/frac_unic_non_iso_valid_mean")
    ap.add_argument("--ratio-col", default="average_ratio_mean")
    ap.add_argument(
        "--comparison", type=int, default=10, metavar="N",
        help="good-vs-bad comparison uses the N best vs N worst configs, ranked "
             "by combined VUN+ratio (default 10)",
    )
    args = ap.parse_args()

    probes, results, results_src = collect(args.paths)
    if args.results:
        results = pd.read_csv(args.results)
        for k in ("eta", "omega"):
            results[k] = results[k].astype(float).round(6)
        results = dedup_results(results)
        results_src = args.results
    objective = pick_objective(results, args.objective)

    out_dir = args.out or (
        args.paths[0] if os.path.isdir(args.paths[0]) else os.path.dirname(args.paths[0])
    )
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = [
        c
        for c in probes.columns
        if c not in NON_FEATURES
        and c not in ("source_file", "n_nodes", "n_pairs")
        and pd.api.types.is_numeric_dtype(probes[c])
    ]
    n_cfg = probes.groupby(JOIN_KEYS).ngroups

    section("1. WHAT YOU HAVE")
    print(f"  probe rows      : {len(probes)}")
    print(f"  configurations  : {n_cfg}")
    print(f"  steps per config: {probes.groupby(JOIN_KEYS)['step'].nunique().unique()}")
    print(f"  results CSV     : {results_src or 'NOT FOUND -- correlations skipped'}")
    print(f"  objective       : {objective or '-'}")
    cfg_tbl = probes.groupby(JOIN_KEYS, as_index=False).size().rename(
        columns={"size": "rows"}
    )
    print("\n" + cfg_tbl.to_string(index=False))

    section("2. WHICH PROBES ARE ALIVE")
    print("  rel_range = (max-min)/|mean| over a trajectory; flat probes cannot")
    print("  rank configurations no matter how good the regressor is.\n")
    diag = probe_diagnostics(probes, feature_cols)
    diag.to_csv(os.path.join(out_dir, "probe_diagnostics.csv"), index=False)
    shown = diag if args.all_features else diag[~diag["probe"].str.endswith("_std")]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not args.all_features:
        n_hidden = len(diag) - len(shown)
        print(f"\n  ({n_hidden} _std columns hidden; --all-features to show, or see")
        print("   probe_diagnostics.csv)")

    section("3. DOES THE SAMPLER SETTLE?")
    print("  final_flip_pct = % of node pairs still flipping at the last step.")
    print("  Model confidence rising while flips stay high = over-stochasticity.\n")
    st = settling_check(probes, results, objective)
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    st.to_csv(os.path.join(out_dir, "probe_settling.csv"), index=False)

    section("4. DENOISER PREDICTION vs SAMPLED STATE")
    print("  Read ACROSS a row: each row compares one statistic computed two ways.")
    print("  jitter    = std(step-to-step change), same units -> raw bounciness")
    print("  roughness = jitter / std(value)     , scale-free -> is one reading")
    print("              trustworthy (1.41 would mean pure noise, 0 a clean trend)\n")
    sh = soft_vs_hard_smoothness(probes)
    if sh.empty:
        print("  no hard_ columns in these logs")
    else:
        print(sh.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        sh.to_csv(os.path.join(out_dir, "probe_soft_vs_hard.csv"), index=False)

    minimize = args.minimize or (
        not args.maximize and objective is not None and "ratio" in objective
    )

    corr, skip_reason = None, None
    if results is None or objective is None:
        skip_reason = "no results CSV / objective found"
    elif n_cfg < 5:
        skip_reason = f"{n_cfg} configuration(s); need >= 5, ideally ~30"
    else:
        per_step = probes.groupby(
            JOIN_KEYS + ["step", "t_frac"], as_index=False
        )[feature_cols].mean()
        merged = per_step.merge(
            results[JOIN_KEYS + [objective]].drop_duplicates(JOIN_KEYS),
            on=JOIN_KEYS,
            how="inner",
        )
        matched = merged[JOIN_KEYS].drop_duplicates().shape[0]
        if matched < 5:
            skip_reason = f"only {matched} configs matched between probes and results"
        else:
            corr = correlate(merged, feature_cols, objective, n_bins=args.bins)
            corr.to_csv(os.path.join(out_dir, "probe_rank_correlation.csv"), index=False)

    section("5. FIGURES")
    written = []

    def add(p):
        if p:
            written.append(p)
            print(f"  {os.path.basename(p)}")

    add(plot_trajectories(probes, results, objective, minimize,
                          os.path.join(out_dir, "1_trajectories.png")))
    add(plot_diagnostics(diag, os.path.join(out_dir, "2_probe_screening.png")))
    add(plot_settling(probes, results, objective, minimize,
                      os.path.join(out_dir, "3_settling.png")))
    add(plot_soft_vs_hard(probes, sh, os.path.join(out_dir, "4_pred_vs_sampled.png")))
    add(plot_landscape(probes, results, objective, minimize,
                       os.path.join(out_dir, "5_search_landscape.png")))

    best_probes = rank_probes(corr, feature_cols, k=3)
    add(plot_probe_vs_quality(probes, results, objective, minimize, best_probes,
                              os.path.join(out_dir, "6_probe_vs_quality.png")))

    top = []
    if corr is not None:
        top = plot_correlation(
            corr, feature_cols, objective,
            os.path.join(out_dir, "7_rank_correlation.png"),
        )
        written.append("7_rank_correlation.png")

    section("6. RANK CORRELATION WITH FINAL QUALITY")
    if corr is None:
        print(f"  SKIPPED: {skip_reason}.")
        print("  Everything above describes trajectory shape only -- it cannot tell")
        print("  you which probe predicts final quality. Run more configs (e.g.")
        print("  sample.search=fixed_configs with a spread of eta/omega/distortor).")
    else:
        print(f"  matched {matched} configurations to results")
        cmp_df = compare_sources(corr, feature_cols)
        if not cmp_df.empty:
            cmp_df.to_csv(
                os.path.join(out_dir, "probe_source_comparison.csv"), index=False
            )
            print("\n  prediction vs sampled state:")
            print(cmp_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\n  earliest trajectory fraction reaching |rho|:")
        print(f"  {'probe':<26}{'>=0.4':>8}{'>=0.7':>8}{'max':>8}")
        for f in top:
            a = corr[["t_bin", f]].dropna()
            hit = lambda thr: (  # noqa: E731
                f"{a.loc[a[f].abs() >= thr, 't_bin'].min():.1f}"
                if (a[f].abs() >= thr).any()
                else "-"
            )
            print(f"  {f:<26}{hit(0.4):>8}{hit(0.7):>8}{a[f].abs().max():>8.2f}")

    section(f"7. GOOD vs BAD (top {args.comparison} vs bottom {args.comparison})")
    if results is None:
        print("  SKIPPED: no results CSV found.")
    else:
        print(f"  comparing the {args.comparison} best vs {args.comparison} worst "
              f"configs, ranked by a combined {args.vun_col} + {args.ratio_col} "
              "score (rest excluded)\n")
        label, quad = label_top_bottom_n(
            results, args.vun_col, args.ratio_col, n=args.comparison
        )
        print(f"  good={quad['good']}  bad={quad['bad']}  "
              f"excluded (in between)={quad['excluded']}")
        if quad["good"] < 2 or quad["bad"] < 2:
            print("\n  SKIPPED plot/table: fewer than 2 configs available on one side.")
        else:
            eff = good_bad_effect_table(probes, feature_cols, label)
            eff.to_csv(os.path.join(out_dir, "probe_good_bad_effect.csv"), index=False)
            print("\n  probes ranked by how well they separate good from bad")
            print("  (cohen_d: |d|>=0.8 large, >=0.5 medium, >=0.2 small; sign shows")
            print("   which direction 'bad' shifts the probe):\n")
            disp = eff[["probe", "best_t", "best_cohen_d"]].head(12)
            print(disp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

            p = plot_good_vs_bad(
                probes, label, quad, os.path.join(out_dir, "8_good_vs_bad.png")
            )
            print(f"\n  wrote {os.path.basename(p)}")

    print(f"\nAll outputs written to {out_dir}\n")


if __name__ == "__main__":
    main()
