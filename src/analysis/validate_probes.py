"""Cross-check every probe in trajectory_probes.py against an independent
reference, driving the actual production code at batch size 1 (so a
per-graph mismatch can't hide behind batch averaging).

  structural probes (density, deg_*, triangles, transitivity, clustering,
  lcc_*, n_components, diameter, aspl, lambda_*):
    - hard graphs -> checked against networkx
    - soft (fractional) graphs -> networkx can't represent them, so checked
      against a from-scratch numpy re-derivation instead

  confidence/movement probes (entropy_e, commit_e, disagree_*, churn, flips,
  jaccard): not graph-theoretic, so checked against a plain-Python
  nested-loop re-derivation of the same definition.

Run:  PYTHONPATH=. python analysis/validate_probes.py
"""

import math
import os
import statistics
import sys

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.trajectory_probes import TrajectoryProbe as TP  # noqa: E402

TOL = 1e-3
rng = np.random.default_rng(0)


class Check:
    def __init__(self):
        self.rows = {}  # name -> [max_abs_diff, n_checked, category]

    def record(self, name, category, got, ref):
        if got is None or ref is None:
            return
        if isinstance(got, float) and math.isnan(got) and isinstance(ref, float) and math.isnan(ref):
            diff = 0.0
        elif (isinstance(got, float) and math.isnan(got)) or (isinstance(ref, float) and math.isnan(ref)):
            diff = float("inf")  # one nan, one not: a real mismatch
        else:
            diff = abs(got - ref)
        key = (name, category)  # hard/soft comparisons must not be averaged together
        row = self.rows.setdefault(key, [0.0, 0])
        row[0] = max(row[0], diff)
        row[1] += 1

    def report(self):
        print(f"{'probe':16s} {'category':30s} {'n':>5} {'max |diff|':>12}  verdict")
        print("-" * 74)
        n_fail = 0
        for (name, cat), (maxdiff, n) in sorted(self.rows.items()):
            ok = maxdiff < TOL
            n_fail += not ok
            print(f"{name:16s} {cat:30s} {n:5d} {maxdiff:12.2e}  {'ok' if ok else 'FAIL'}")
        print()
        print("ALL PASS" if n_fail == 0 else f"{n_fail} PROBE/CATEGORY PAIR(S) MISMATCHED")
        return n_fail


check = Check()

# --------------------------------------------------------------------------
# Graph corpus
# --------------------------------------------------------------------------

def hard_graphs():
    """(networkx.Graph, adjacency ndarray) pairs -- the ground truth is real."""
    out = []
    for p in (0.0, 0.05, 0.15, 0.35, 0.7, 1.0):
        for _ in range(6):
            n = int(rng.integers(3, 14))
            g = nx.gnp_random_graph(n, p, seed=int(rng.integers(1e6)))
            out.append(g)
    for name, g in [
        ("path", nx.path_graph(10)), ("star", nx.star_graph(9)),
        ("cycle", nx.cycle_graph(9)), ("complete", nx.complete_graph(7)),
        ("empty", nx.empty_graph(6)), ("grid", nx.grid_2d_graph(3, 3)),
        ("tree", nx.random_tree(11, seed=2)),
        ("two_components", nx.disjoint_union(nx.path_graph(5), nx.cycle_graph(4))),
        ("with_isolates", nx.disjoint_union(nx.star_graph(5), nx.empty_graph(3))),
    ]:
        out.append(nx.convert_node_labels_to_integers(g))
    return out


def soft_graphs():
    """Fractional adjacency matrices -- no networkx equivalent, numpy only."""
    out = []
    for _ in range(20):
        n = int(rng.integers(3, 14))
        a = rng.random((n, n))
        a = np.triu(a, 1)
        a = a + a.T
        out.append(a)
    return out


# --------------------------------------------------------------------------
# References for structural probes
# --------------------------------------------------------------------------

def nx_reference(g, n):
    degs = [d for _, d in g.degree()] or [0]
    comps = list(nx.connected_components(g))
    lcc = max((len(c) for c in comps), default=0)
    dists = [
        L for _, lengths in nx.all_pairs_shortest_path_length(g)
        for dst, L in lengths.items() if L > 0
    ]
    return {
        "density": nx.density(g) if n > 1 else float("nan"),
        "deg_mean": statistics.mean(degs),
        "deg_var": statistics.pvariance(degs),
        "deg_max": max(degs),
        "triangles": sum(nx.triangles(g).values()) // 3,
        "transitivity": nx.transitivity(g),
        "clustering": nx.average_clustering(g),
        "lcc_size": lcc,
        "lcc_frac": lcc / n if n else float("nan"),
        "n_components": len(comps),
        "diameter": max(dists) if dists else float("nan"),
        "aspl": statistics.mean(dists) if dists else float("nan"),
        "lambda_max": float(np.linalg.eigvalsh(nx.to_numpy_array(g))[-1]) if n else float("nan"),
        "lambda_min": float(np.linalg.eigvalsh(nx.to_numpy_array(g))[0]) if n else float("nan"),
    }


def numpy_reference_soft(a, n):
    """Independent re-derivation from the math (e.g. triangles via nested
    loops over triples, not trace(A^3)/6), so a bug shared with
    trajectory_probes.py would have to be a coincidence, not a copy-paste."""
    deg = a.sum(-1)
    deg_mean = deg.mean()
    deg_var = ((deg - deg_mean) ** 2).mean()
    deg_max = deg.max()
    n_pairs = n * (n - 1)
    density = a.sum() / n_pairs if n_pairs else float("nan")

    tri_sum, wedge_sum, clust_sum = 0.0, 0.0, 0.0
    for i in range(n):
        wedges_i = 0.0
        tri_i = 0.0
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                wedges_i += a[i, j] * a[i, k]
                tri_i += a[i, j] * a[j, k] * a[i, k]
        wedge_sum += wedges_i
        tri_sum += tri_i
        clust_sum += tri_i / wedges_i if wedges_i > 1e-9 else 0.0
    triangles = tri_sum / 6.0
    transitivity = tri_sum / wedge_sum if wedge_sum > 1e-9 else 0.0
    clustering = clust_sum / n

    # Plain BFS at the same 0.5 threshold the probe uses.
    adj01 = a > 0.5
    seen_all = []
    for s in range(n):
        dist = {s: 0}
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in range(n):
                    if adj01[u, v] and v not in dist:
                        dist[v] = d
                        nxt.append(v)
            frontier = nxt
        seen_all.append(dist)
    comp_of = [-1] * n
    cid = 0
    for s in range(n):
        if comp_of[s] != -1:
            continue
        for v in seen_all[s]:
            comp_of[v] = cid
        cid += 1
    sizes = [comp_of.count(c) for c in range(cid)]
    lcc = max(sizes) if sizes else 0
    dists = [L for s in range(n) for v, L in seen_all[s].items() if L > 0]

    eig = np.linalg.eigvalsh(a)
    return {
        "density": density, "deg_mean": deg_mean, "deg_var": deg_var, "deg_max": deg_max,
        "triangles": triangles, "transitivity": transitivity, "clustering": clustering,
        "lcc_size": lcc, "lcc_frac": lcc / n if n else float("nan"), "n_components": cid,
        "diameter": max(dists) if dists else float("nan"),
        "aspl": statistics.mean(dists) if dists else float("nan"),
        "lambda_max": float(eig[-1]), "lambda_min": float(eig[0]),
    }


def probe_structural(a_np, n):
    """Run the real TrajectoryProbe._adj_stats at batch size 1."""
    A = torch.tensor(a_np[None], dtype=torch.float)
    nm = torch.ones(1, n, dtype=torch.bool)
    eye = torch.eye(n, dtype=torch.bool)
    pair = (nm[:, :, None] & nm[:, None, :]) & ~eye
    n_pairs = pair.sum((-1, -2)).clamp(min=1).float()
    n_nodes = nm.sum(-1).clamp(min=1).float()
    out = TP(spectral=True)._adj_stats(A, nm, n_nodes, n_pairs, step=0, prefix="")
    return {k[:-5]: v for k, v in out.items() if k.endswith("_mean")}


# --------------------------------------------------------------------------
# References for confidence / movement probes (no graph, no networkx)
# --------------------------------------------------------------------------

def confidence_reference(pred_E, cur_cls, n):
    """Plain nested-loop re-derivation; matches the clamp(1e-9) log semantics
    of trajectory_probes.py without calling it."""
    ent_sum = disagree_sum = commit_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count += 1
            p0, p1 = float(pred_E[i, j, 0]), float(pred_E[i, j, 1])
            ent_sum += -(math.log(max(p0, 1e-9)) * p0 + math.log(max(p1, 1e-9)) * p1)
            commit_sum += float(max(p0, p1) > 0.9)
            p_cur = p0 if cur_cls[i, j] == 0 else p1
            disagree_sum += 1.0 - p_cur
    return {
        "entropy_e": ent_sum / count, "commit_e": commit_sum / count,
        "disagree_frac": disagree_sum / count, "disagree_count": disagree_sum / 2.0,
    }


def movement_reference(A_cur, A_prev, cls_cur, cls_prev, n):
    churn = inter = union = 0.0
    hinter = humion = flips = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count += 1
            churn += abs(A_cur[i, j] - A_prev[i, j])
            inter += min(A_cur[i, j], A_prev[i, j])
            union += max(A_cur[i, j], A_prev[i, j])
            h_cur = float(cls_cur[i, j] != 0)
            h_prev = float(cls_prev[i, j] != 0)
            hinter += min(h_cur, h_prev)
            humion += max(h_cur, h_prev)
            flips += float(cls_cur[i, j] != cls_prev[i, j])
    return {
        "churn": churn / count,
        "flips": flips / 2.0,
        "jaccard": (inter / union) if union > 0 else 1.0,
        "hard_jaccard": (hinter / humion) if humion > 0 else 1.0,
    }


def probe_confidence_and_movement(pred_E0, E_t0, pred_X0, pred_E1, E_t1, pred_X1, n):
    """Run the real TrajectoryProbe.record() twice (to populate 'previous
    step' state) at batch size 1, and read back what it logged."""
    p = TP(spectral=False)
    nm = torch.ones(1, n, dtype=torch.bool)
    p.record(X_t=torch.nn.functional.one_hot(pred_X0.argmax(-1), pred_X0.shape[-1]).float(),
              E_t=E_t0, pred_X=pred_X0, pred_E=pred_E0, node_mask=nm)
    p.record(X_t=torch.nn.functional.one_hot(pred_X1.argmax(-1), pred_X1.shape[-1]).float(),
              E_t=E_t1, pred_X=pred_X1, pred_E=pred_E1, node_mask=nm)
    row = p._rows[-1]
    return {k: row[f"{k}_mean"] for k in
            ("entropy_e", "commit_e", "disagree_frac", "disagree_count",
             "churn", "flips", "jaccard", "hard_jaccard")}


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------

print("=== structural probes: hard graphs vs networkx ===")
for g in hard_graphs():
    n = g.number_of_nodes()
    if n == 0:
        continue
    a = nx.to_numpy_array(g)
    got = probe_structural(a, n)
    ref = nx_reference(g, n)
    for k in ref:
        check.record(k, "networkx (hard graph)", got.get(k), ref[k])

print("=== structural probes: soft graphs vs from-scratch numpy ===")
for a in soft_graphs():
    n = a.shape[0]
    got = probe_structural(a, n)
    ref = numpy_reference_soft(a, n)
    for k in ref:
        check.record(k, "numpy re-derivation (soft)", got.get(k), ref[k])

print("=== confidence / movement probes: plain-Python re-derivation ===")
for _ in range(25):
    n = int(rng.integers(3, 12))
    def rand_pred_e():
        logits = torch.randn(1, n, n, 2) * 2
        pe = torch.softmax(logits, -1)
        return 0.5 * (pe + pe.transpose(1, 2))
    def rand_e_t(n):
        cls = torch.tensor((rng.random((n, n)) < 0.3).astype(int))
        cls = torch.triu(cls, 1); cls = cls + cls.T
        cls = cls.unsqueeze(0)  # (1, n, n) -- record() expects a batch dim
        return torch.nn.functional.one_hot(cls, 2).float(), cls
    pred_E0, pred_E1 = rand_pred_e(), rand_pred_e()
    E_t0, cls0 = rand_e_t(n)
    E_t1, cls1 = rand_e_t(n)
    pred_X0 = torch.softmax(torch.randn(1, n, 3), -1)
    pred_X1 = torch.softmax(torch.randn(1, n, 3), -1)

    got = probe_confidence_and_movement(pred_E0, E_t0, pred_X0, pred_E1, E_t1, pred_X1, n)
    ref_c = confidence_reference(pred_E1[0].numpy(), cls1[0].numpy(), n)
    A0 = (1 - pred_E0[0, ..., 0].numpy())
    A1 = (1 - pred_E1[0, ..., 0].numpy())
    ref_m = movement_reference(A1, A0, cls1[0].numpy(), cls0[0].numpy(), n)
    for k, v in {**ref_c, **ref_m}.items():
        check.record(k, "plain-Python loop (no networkx)", got.get(k), v)

print()
n_fail = check.report()
sys.exit(1 if n_fail else 0)
