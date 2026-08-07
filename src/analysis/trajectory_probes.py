"""Cheap per-step structural probes over the sampling trajectory."""

import os

import numpy as np
import pandas as pd
import torch


def _resolve_validity_fn(dataset_name):
    name = str(dataset_name).lower()
    try:
        from analysis import spectre_utils as su
    except Exception:
        return None, None

    if "planar" in name:
        return "planar", su.is_planar_graph
    if "sbm" in name:
        # strict=False keeps the refinement cheap; we only need a rank signal
        return "sbm", lambda g: su.is_sbm_graph(g, strict=False, refinement_steps=10)
    if "tree" in name:
        import networkx as nx

        return "tree", lambda g: nx.is_tree(g)
    if "lobster" in name:
        return "lobster", su.is_lobster_graph
    if "grid" in name:
        return "grid", su.is_grid_graph
    return None, None


class TrajectoryProbe:

    FILENAME = "trajectory_probes.csv"

    def __init__(
        self,
        dataset_name=None,
        spectral=True,
        validity_every=0,
        validity_max_graphs=8,
    ):
        self.spectral = spectral
        self.validity_every = int(validity_every)
        self.validity_max_graphs = int(validity_max_graphs)
        self.validity_name, self.validity_fn = _resolve_validity_fn(dataset_name)

        self.output_dir = os.getcwd()
        self._rows = []
        self._context = {}
        self._trial_seq = -1
        self._wrote_header = False
        self._reset_batch_state()

    # ------------------------------------------------------------------ setup

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @property
    def csv_path(self):
        return os.path.join(self.output_dir, self.FILENAME)

    def _reset_batch_state(self):
        self._step = 0
        self._batch_id = 0
        self._sample_steps = None
        self._prev_A = None
        self._prev_H = None
        self._prev_cls = None

    # -------------------------------------------------------------- lifecycle

    def begin_trial(self, **context):
        self._trial_seq += 1
        self._context = dict(context)
        self._context["trial_seq"] = self._trial_seq
        self._reset_batch_state()

    def begin_batch(self, batch_id, sample_steps):
        self._step = 0
        self._batch_id = int(batch_id)
        self._sample_steps = int(sample_steps)
        self._prev_A = None
        self._prev_H = None
        self._prev_cls = None

    def end_trial(self):
        self.flush()

    def flush(self):
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        header = not (self._wrote_header or os.path.exists(self.csv_path))
        df.to_csv(self.csv_path, mode="a", header=header, index=False)
        self._wrote_header = True
        self._rows = []

    # ---------------------------------------------------------------- probing

    @torch.no_grad()
    def record(self, X_t, E_t, pred_X, pred_E, node_mask, t=None, extra_y=None):
        """Log one row of statistics for the current sampling step.
        Called from sample_p_zs_given_zt, after pred_X/pred_E are computed and
        before the conditional branch overwrites them.
        """
        step = self._step
        self._step += 1

        nm = node_mask.bool()
        bs, n = nm.shape
        eye = torch.eye(n, device=nm.device, dtype=torch.bool)
        pair = (nm[:, :, None] & nm[:, None, :]) & ~eye  # (bs, n, n)
        n_pairs = pair.sum((-1, -2)).clamp(min=1).float()
        n_nodes = nm.sum(-1).clamp(min=1).float()

        pf = pair.float()

        # -- soft adjacency: the denoiser's expected final graph --------------
        A = (1.0 - pred_E[..., 0]) * pf
        A = 0.5 * (A + A.transpose(-1, -2))  # defensive symmetrisation

        # -- hard adjacency: the state the rate matrix actually sampled -------
        cur_cls = E_t.argmax(-1)  # (bs, n, n)
        H = ((cur_cls != 0) & pair).float()

        # -- how much the model still disagrees with the current state --------
        p_cur = pred_E.gather(-1, cur_cls.unsqueeze(-1)).squeeze(-1)
        disagree_frac = ((1.0 - p_cur) * pf).sum((-1, -2)) / n_pairs
        disagree_count = ((1.0 - p_cur) * pf).sum((-1, -2)) / 2.0

        # -- model confidence -------------------------------------------------
        ent_e = -(pred_E.clamp_min(1e-9).log() * pred_E).sum(-1)
        ent_e = (ent_e * pf).sum((-1, -2)) / n_pairs
        commit_e = ((pred_E.max(-1).values > 0.9).float() * pf).sum((-1, -2)) / n_pairs

        ent_x = -(pred_X.clamp_min(1e-9).log() * pred_X).sum(-1)
        ent_x = (ent_x * nm).sum(-1) / n_nodes

        # -- trajectory dynamics: churn/flips = how much moved; jaccard = how
        # much of the edge set survived (scale-free, unlike churn/flips) -----
        if self._prev_A is not None:
            churn = (A - self._prev_A).abs().sum((-1, -2)) / n_pairs
            flips = ((cur_cls != self._prev_cls) & pair).float().sum((-1, -2)) / 2.0
            jaccard = self._jaccard(A, self._prev_A)
            hard_jaccard = self._jaccard(H, self._prev_H)
        else:
            churn = torch.zeros_like(n_pairs)
            flips = torch.zeros_like(n_pairs)
            jaccard = torch.full_like(n_pairs, float("nan"))  # step 0: no prev step
            hard_jaccard = torch.full_like(n_pairs, float("nan"))
        self._prev_A = A
        self._prev_H = H
        self._prev_cls = cur_cls

        row = {
            "step": step,
            "t_frac": (step / self._sample_steps) if self._sample_steps else np.nan,
            "t_distorted": float(t[0].item()) if t is not None else np.nan,
            "batch_id": self._batch_id,
            "n_nodes": float(n_nodes.mean().item()),
            "n_pairs": float(n_pairs.mean().item()) / 2.0,
        }
        row.update(self._context)

        row.update(self._adj_stats(A, nm, n_nodes, n_pairs, step, prefix=""))
        row.update(self._adj_stats(H, nm, n_nodes, n_pairs, step, prefix="hard_"))

        for key, val in (
            ("disagree_frac", disagree_frac),
            ("disagree_count", disagree_count),
            ("entropy_e", ent_e),
            ("entropy_x", ent_x),
            ("commit_e", commit_e),
            ("churn", churn),
            ("flips", flips),
            ("jaccard", jaccard),
            ("hard_jaccard", hard_jaccard),
        ):
            row.update(self._agg(key, val))

        # -- graph-level features ExtraFeatures already computed for free -----
        if extra_y is not None and extra_y.numel():
            xf = extra_y.detach().float().mean(0).cpu().numpy()
            for i, v in enumerate(xf):
                row[f"xf_{i}"] = float(v)

        # -- MAP validity: "if I stopped now, would this graph be valid?" -----
        if self.validity_every and step % self.validity_every == 0:
            row.update(self._map_graph_stats(pred_E, nm, pair))

        self._rows.append(row)

    def _adj_stats(self, A, nm, n_nodes, n_pairs, step, prefix=""):
        """Structural statistics of one adjacency matrix (soft or hard -- same code)."""
        deg = A.sum(-1)
        deg_mean = (deg * nm).sum(-1) / n_nodes
        deg_var = (((deg - deg_mean[:, None]) ** 2) * nm).sum(-1) / n_nodes
        deg_max = (deg - (~nm) * 1e9).max(-1).values
        density = A.sum((-1, -2)) / n_pairs

        tri_node = ((A @ A) * A).sum(-1)  # diag(A^3)[i] = 2 * triangles through i
        triangles = tri_node.sum(-1) / 6.0

        wedges_node = (deg**2 - (A**2).sum(-1)).clamp(min=0.0)
        transitivity = tri_node.sum(-1) / wedges_node.sum(-1).clamp(min=1e-9)
        clust_node = tri_node / wedges_node.clamp(min=1e-9)  # per-node avg, not ratio-of-sums
        clustering = (clust_node * nm).sum(-1) / n_nodes

        lcc_size, lcc_frac, n_components, diameter, aspl = self._reach_stats(
            A, nm, n_nodes
        )

        out = {}
        for key, val in (
            ("density", density),
            ("deg_mean", deg_mean),
            ("deg_var", deg_var),
            ("deg_max", deg_max),
            ("triangles", triangles),
            ("transitivity", transitivity),
            ("clustering", clustering),
            ("lcc_size", lcc_size),
            ("lcc_frac", lcc_frac),
            ("n_components", n_components),
            ("diameter", diameter),
            ("aspl", aspl),
        ):
            out.update(self._agg(prefix + key, val))

        if self.spectral:
            try:
                eigvals = torch.linalg.eigvalsh(A.float())
                out.update(self._agg(prefix + "lambda_max", eigvals[..., -1]))
                out.update(self._agg(prefix + "lambda_min", eigvals[..., 0]))
            except Exception as exc:  # pragma: no cover - numerical edge cases
                out[prefix + "lambda_max_mean"] = np.nan
                out[prefix + "lambda_min_mean"] = np.nan
                if step == 0:
                    print(f"[trajectory_probe] eigvalsh failed, skipping: {exc}")
        return out

    # Threshold for "edge exists" on the soft adjacency; a no-op on the hard one.
    EDGE_THRESHOLD = 0.5

    MAX_BFS_ITERS = 64

    @classmethod
    def _reach_stats(cls, A, nm, n_nodes):
        """Components *and* shortest-path statistics, batched, no networkx."""
        n = A.shape[-1]
        eye = torch.eye(n, device=A.device, dtype=torch.bool)
        valid = nm.bool()
        pair = (valid[:, :, None] & valid[:, None, :]) & ~eye

        # Padded nodes dropped from the closure (else each is its own component).
        pair_f = pair.float()
        adj = ((A > cls.EDGE_THRESHOLD) & pair).float()
        reach = (eye & valid[:, :, None]).float()
        dist = torch.zeros_like(adj)

        incomplete = torch.zeros(A.shape[0], dtype=torch.bool, device=A.device)
        growing = incomplete
        # n - 1 levels is enough for any pair; +1 to detect the "done" break.
        max_iters = min(max(n, 2), cls.MAX_BFS_ITERS)
        for k in range(1, max_iters + 1):
            nxt = torch.clamp(torch.bmm(reach, adj) + reach, max=1.0)
            newly = nxt - reach  # 1 exactly on the pairs newly reached at distance k
            growing = newly.sum((-1, -2)) > 0  # (bs,) still expanding?
            if not bool(growing.any()):
                break
            dist = dist + float(k) * newly
            reach = nxt
        else:
            # Still expanding at the cap: distances truncate, but finish the
            # closure by squaring so components stay exact regardless.
            incomplete = growing
            for _ in range(max(1, int(np.ceil(np.log2(max(n, 2)))))):
                reach = torch.clamp(torch.bmm(reach, reach), max=1.0)

        comp_size = reach.sum(-1)  # (bs, n), 0 for padded rows
        lcc_size = (comp_size * valid).max(-1).values
        lcc_frac = lcc_size / n_nodes
        n_components = (valid.float() / comp_size.clamp(min=1.0)).sum(-1)

        # Unreachable pairs contribute 0, so they raise neither the max nor the
        # sum; the mean is over the reachable pairs only.
        connected = reach * pair_f
        n_connected = connected.sum((-1, -2))
        d = dist * connected
        diameter = d.flatten(1).max(-1).values
        aspl = d.sum((-1, -2)) / n_connected.clamp(min=1.0)

        defined = (n_connected > 0) & ~incomplete
        nan = torch.full_like(aspl, float("nan"))
        diameter = torch.where(defined, diameter, nan)
        aspl = torch.where(defined, aspl, nan)
        return lcc_size, lcc_frac, n_components, diameter, aspl

    @staticmethod
    def _jaccard(cur, prev):
        """Weighted Jaccard sum(min)/sum(max); degenerate 0/0 -> 1.0 (unchanged)."""
        inter = torch.minimum(cur, prev).sum((-1, -2))
        union = torch.maximum(cur, prev).sum((-1, -2))
        return torch.where(union > 0, inter / union.clamp(min=1e-9),
                           torch.ones_like(union))

    @staticmethod
    def _agg(name, values):
        """Batch mean/std, nan-aware: a stat undefined for some graphs (no
        diameter on an edgeless graph, no Jaccard at step 0) drops out instead
        of poisoning the rest of the batch."""
        v = values.detach().float()
        ok = torch.isfinite(v)
        if not bool(ok.any()):
            return {f"{name}_mean": np.nan, f"{name}_std": np.nan}
        v = v[ok]
        return {
            f"{name}_mean": float(v.mean().item()),
            f"{name}_std": float(v.std(unbiased=False).item()),
        }

    def _map_graph_stats(self, pred_E, nm, pair):
        """Hard checks on argmax(pred_E) -- the MAP graph at the current step."""
        import networkx as nx

        adj = (pred_E.argmax(-1) != 0) & pair
        k = min(self.validity_max_graphs, adj.shape[0])
        adj = adj[:k].cpu().numpy()
        sizes = nm[:k].sum(-1).cpu().numpy()

        connected, valid = [], []
        for a, n_i in zip(adj, sizes):
            g = nx.from_numpy_array(a[: int(n_i), : int(n_i)].astype(np.uint8))
            connected.append(float(nx.is_connected(g)) if g.number_of_nodes() else 0.0)
            if self.validity_fn is not None:
                try:
                    valid.append(float(bool(self.validity_fn(g))))
                except Exception:
                    valid.append(np.nan)

        out = {"map_connected_frac": float(np.mean(connected)) if connected else np.nan}
        if valid:
            out["map_valid_frac"] = float(np.nanmean(valid))
            out["map_valid_criterion"] = self.validity_name
        return out
