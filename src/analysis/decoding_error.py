"""Decoding error rate P_e(t) of a trained denoiser: how much of the graph is
already determined at noise level t. Real graph -> corrupt to level t -> one
forward pass -> score argmax(pred) against truth; no sampling involved.

Noise is coupled across t by default: one (U, R) draw per token, state at t
keeps the truth where U < t, so revealed sets nest as t grows and the curve
stays close to monotone. Three free error signals per pass:

    hard   1[argmax != truth]           noisiest
    soft   1 - p_theta(truth | G_t)     bounded, usually cleanest
    ce     -log p_theta(truth | G_t)    unbounded as t -> 0, clipped
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import utils


def _symmetrize_upper(M):
    """Mirror the strict upper triangle onto the lower one; zero the diagonal."""
    M = torch.triu(M, diagonal=1)
    return M + M.transpose(1, 2)


def _sample_from_limit(limit, shape, device, generator=None):
    """Draw class labels of the given shape i.i.d. from a limit distribution."""
    probs = limit.to(device)[None, :].expand(int(np.prod(shape)), -1)
    labels = torch.multinomial(probs, 1, replacement=True, generator=generator)
    return labels.reshape(shape)


@torch.no_grad()
def measure_decoding_error(
    model,
    dataloader,
    t_grid,
    n_draws=8,
    coupled=True,
    max_graphs=None,
    seed=0,
    ce_clip=20.0,
    verbose=True,
):
    """Measure P_e(t) for a trained model over a grid of noise levels.
    Returns a DataFrame with one row per t."""
    device = next(model.parameters()).device
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_t = len(t_grid)

    dims = model.noise_dist.get_noise_dims()
    lim_X = model.limit_dist.X.to(device).float()
    lim_E = model.limit_dist.E.to(device).float()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    acc = {
        k: np.zeros(n_t, dtype=np.float64)
        for k in (
            "err_X", "soft_X", "ce_X", "cnt_X",
            "err_E", "soft_E", "ce_E", "cnt_E",
        )
    }

    n_graphs = 0
    n_forward = 0
    t0 = time.time()

    for batch_idx, data in enumerate(dataloader):
        if max_graphs is not None and n_graphs >= max_graphs:
            break

        data = data.to(device)
        if data.edge_index.numel() == 0:
            continue

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)

        # Ground truth -- untouched from here on.
        X1 = dense_data.X.argmax(dim=-1)  # (b, n)
        E1 = dense_data.E.argmax(dim=-1)  # (b, n, n)
        bs, n = X1.shape

        node_ok = node_mask  # (b, n) bool
        pair_ok = node_mask[:, :, None] & node_mask[:, None, :]
        pair_ok = pair_ok & torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1
        )[None]

        for _ in range(n_draws):
            if coupled:
                U_X, R_X, U_E, R_E = _draw_corruption(
                    bs, n, dims, lim_X, lim_E, device, gen
                )

            for i_t, t_val in enumerate(t_grid):
                if not coupled:
                    U_X, R_X, U_E, R_E = _draw_corruption(
                        bs, n, dims, lim_X, lim_E, device, gen
                    )

                # Keep the truth where U < t, else the replacement.
                Xt_lab = torch.where(U_X < t_val, X1, R_X)
                Et_lab = torch.where(U_E < t_val, E1, R_E)

                t_tensor = torch.full((bs, 1), float(t_val), device=device)
                noisy_data = {
                    "t": t_tensor,
                    "X_t": F.one_hot(Xt_lab, num_classes=dims["X"]).float(),
                    "E_t": F.one_hot(Et_lab, num_classes=dims["E"]).float(),
                    "y_t": data.y,
                    "node_mask": node_mask,
                }
                z_t = utils.PlaceHolder(
                    X=noisy_data["X_t"], E=noisy_data["E_t"], y=data.y
                ).mask(node_mask)
                noisy_data["X_t"], noisy_data["E_t"] = z_t.X, z_t.E

                extra_data = model.compute_extra_data(noisy_data)
                pred = model.forward(noisy_data, extra_data, node_mask)
                n_forward += 1

                p_X = F.softmax(pred.X, dim=-1)
                p_E = F.softmax(pred.E, dim=-1)

                _accumulate(acc, i_t, "X", p_X, X1, node_ok, ce_clip)
                _accumulate(acc, i_t, "E", p_E, E1, pair_ok, ce_clip)

        n_graphs += bs
        if verbose:
            print(
                f"[decoding_error] batch {batch_idx}: {n_graphs} graphs, "
                f"{n_forward} forwards, {time.time() - t0:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(
        {
            "t": t_grid,
            "pe_X": acc["err_X"] / np.maximum(acc["cnt_X"], 1),
            "pe_E": acc["err_E"] / np.maximum(acc["cnt_E"], 1),
            "soft_X": acc["soft_X"] / np.maximum(acc["cnt_X"], 1),
            "soft_E": acc["soft_E"] / np.maximum(acc["cnt_E"], 1),
            "ce_X": acc["ce_X"] / np.maximum(acc["cnt_X"], 1),
            "ce_E": acc["ce_E"] / np.maximum(acc["cnt_E"], 1),
            "n_tokens_X": acc["cnt_X"],
            "n_tokens_E": acc["cnt_E"],
        }
    )
    df.attrs["n_graphs"] = n_graphs
    df.attrs["n_draws"] = n_draws
    df.attrs["coupled"] = coupled
    return df


def _draw_corruption(bs, n, dims, lim_X, lim_E, device, gen):
    """One uniform field and one replacement field, for nodes and edges."""
    U_X = torch.rand((bs, n), device=device, generator=gen)
    R_X = _sample_from_limit(lim_X, (bs, n), device, gen)

    # Draw the upper triangle and mirror it, so the corrupted state stays
    # symmetric at every t.
    U_E = _symmetrize_upper(torch.rand((bs, n, n), device=device, generator=gen))
    R_E = _symmetrize_upper(_sample_from_limit(lim_E, (bs, n, n), device, gen))
    return U_X, R_X, U_E, R_E


def _accumulate(acc, i_t, key, probs, truth, ok, ce_clip):
    """Add this batch's hard / soft / CE errors into the slot for this t."""
    n_classes = probs.shape[-1]
    truth_c = truth.clamp(max=n_classes - 1)

    wrong = (probs.argmax(dim=-1) != truth) & ok
    p_true = probs.gather(-1, truth_c.unsqueeze(-1)).squeeze(-1)
    ce = (-torch.log(p_true.clamp_min(1e-30))).clamp(max=ce_clip)

    acc[f"err_{key}"][i_t] += wrong.sum().item()
    acc[f"soft_{key}"][i_t] += ((1.0 - p_true) * ok).sum().item()
    acc[f"ce_{key}"][i_t] += (ce * ok).sum().item()
    acc[f"cnt_{key}"][i_t] += ok.sum().item()


def combine_modalities(df, lambda_E):
    """Weighted node+edge curve -> single curve the schedule needs; node
    curve dropped if trivial (single node class, e.g. planar/SBM)."""
    node_is_trivial = bool(np.allclose(df["pe_X"].values, 0.0))
    for signal in ("pe", "soft", "ce"):
        if node_is_trivial:
            df[f"{signal}_combined"] = df[f"{signal}_E"]
        else:
            df[f"{signal}_combined"] = (
                df[f"{signal}_X"] + lambda_E * df[f"{signal}_E"]
            ) / (1.0 + lambda_E)
    df.attrs["node_is_trivial"] = node_is_trivial
    df.attrs["lambda_E"] = lambda_E
    return df


def describe_curve(df, signal="pe_combined"):
    """Human-readable summary: where does the decoding actually happen?"""
    t = df["t"].values
    y = df[signal].values
    hi, lo = y[0], y[-1]
    span = hi - lo
    lines = [
        f"signal={signal}  P_e(0)={hi:.4f}  P_e(1)={lo:.4f}  span={span:.4f}",
    ]
    if span <= 1e-9:
        lines.append("  curve is flat -- no usable schedule can be derived from it")
        return "\n".join(lines)

    tau = (hi - y) / span
    for frac in (0.25, 0.5, 0.75, 0.9):
        idx = int(np.searchsorted(np.maximum.accumulate(tau), frac))
        idx = min(idx, len(t) - 1)
        lines.append(f"  {int(frac * 100)}% of the decoding done by t = {t[idx]:.3f}")

    quarters = []
    for q in range(4):
        a = np.searchsorted(t, q * 0.25)
        b = min(np.searchsorted(t, (q + 1) * 0.25), len(t) - 1)
        quarters.append((y[a] - y[b]) / span)
    lines.append(
        "  share of decoding per quarter of t: "
        + ", ".join(f"[{q * 0.25:.2f},{(q + 1) * 0.25:.2f}]={v:.1%}"
                    for q, v in enumerate(quarters))
    )
    return "\n".join(lines)


def plot_curves(df, out_path):
    """Plot every measured signal against t. Optional; skipped if no matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[decoding_error] skipping plot: {e}")
        return

    signals = [c for c in ("pe_combined", "soft_combined", "ce_combined") if c in df]
    fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 4))
    axes = np.atleast_1d(axes)
    for ax, sig in zip(axes, signals):
        ax.plot(df["t"], df[sig], lw=2)
        ax.set_xlabel("t")
        ax.set_ylabel(sig)
        ax.set_title(sig)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[decoding_error] plot -> {out_path}")


def run_and_save(
    model,
    dataloader,
    out_dir,
    n_times=51,
    n_draws=8,
    coupled=True,
    max_graphs=None,
    seed=0,
    lambda_E=5.0,
):
    """Measure, combine, describe, and write curve.csv + curve.png to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    t_grid = np.linspace(0.0, 1.0, n_times)

    df = measure_decoding_error(
        model,
        dataloader,
        t_grid,
        n_draws=n_draws,
        coupled=coupled,
        max_graphs=max_graphs,
        seed=seed,
    )
    df = combine_modalities(df, lambda_E)

    csv_path = os.path.join(out_dir, "decoding_error.csv")
    df.to_csv(csv_path, index=False)
    print(f"[decoding_error] curve -> {csv_path}")

    plot_curves(df, os.path.join(out_dir, "decoding_error.png"))

    summary = "\n".join(
        [
            f"graphs={df.attrs['n_graphs']}  draws={df.attrs['n_draws']}  "
            f"coupled={df.attrs['coupled']}  n_times={n_times}",
            f"node curve trivial (single node class): {df.attrs['node_is_trivial']}",
            describe_curve(df, "pe_combined"),
            describe_curve(df, "soft_combined"),
            _monotonicity_report(df),
        ]
    )
    print(summary)
    with open(os.path.join(out_dir, "decoding_error_summary.txt"), "w") as f:
        f.write(summary + "\n")

    return df


def _monotonicity_report(df):
    """P_e must be non-increasing in t; violations are Monte-Carlo noise."""
    lines = []
    for sig in ("pe_combined", "soft_combined"):
        y = df[sig].values
        d = np.diff(y)
        n_viol = int((d > 0).sum())
        worst = float(d.max()) if len(d) else 0.0
        lines.append(
            f"{sig}: {n_viol}/{len(d)} increasing steps, largest increase {worst:+.5f}"
        )
    return "\n".join(lines)
