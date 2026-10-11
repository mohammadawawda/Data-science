"""Null models: are the detected communities real structure or artifacts?

Null A (degrees kept, wiring random): every country keeps exactly its number
of outgoing and incoming corridors; who it connects to is rewired by directed
edge swaps (a->b, c->d  =>  a->d, c->b; two swaps per edge), and the real
weights are then dealt out at random over the rewired edges. This is the
configuration-model null from class on our directed weighted graphs: degree
heterogeneity survives, geography / history / volume structure does not.
The earlier size-only multinomial null is gone. That network was nearly
complete with weights that factor into origin x destination totals, so
Infomap never found more than one module in it and the comparison was empty.

Null B (topology kept, weights shuffled; 2015 only): real corridor set,
permuted volumes. Separates which corridors exist from how much flows.

100 replicates per period, Infomap at the same num_trials as the observed
partition (trial parity). Per period we report null mean / sd / max, z-score,
empirical p = (#null >= observed + 1) / (n + 1), and excess savings =
observed - null mean, i.e. structure beyond what degrees alone give you.
Runtime: ~2-3 s per replicate, ~30 min for the whole stage.
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi
from common import load_flows, build_graphs, bloc_labels, PERIODS, OUT, EST
from run_communities import run_infomap, score_vs_blocs

N_REPS = int(os.environ.get("MIGRATION_NULL_REPS", 100))  # lower for a smoke test
NUM_TRIALS = 20          # same as the observed Infomap run
SWAPS_PER_EDGE = 2       # 2m accepted swaps fully randomizes the wiring
SEED = 7


def rewire_degree_preserving(G, swaps_per_edge=SWAPS_PER_EDGE, rng=None):
    """Directed edge swaps that keep every in- and out-degree and the edge
    count. A proposal (a->b, c->d) -> (a->d, c->b) is rejected if it would
    create a self-loop or an edge that already exists. Our graphs are dense
    (2015: 66% of possible corridors exist), so most proposals are rejected;
    we draw a big batch of candidate pairs up front and stop once
    swaps_per_edge * m swaps went through. Returns (edge list, swaps done)."""
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    src = [idx[u] for u, _ in G.edges()]
    dst = [idx[v] for _, v in G.edges()]
    m = len(src)
    adj = [bytearray(len(nodes)) for _ in nodes]
    for u, v in zip(src, dst):
        adj[u][v] = 1
    target = swaps_per_edge * m
    ii = rng.integers(m, size=160 * m).tolist()
    jj = rng.integers(m, size=160 * m).tolist()
    done = 0
    for i, j in zip(ii, jj):
        if done >= target:
            break
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if a == d or c == b or adj[a][d] or adj[c][b]:
            continue
        adj[a][b] = 0; adj[c][d] = 0; adj[a][d] = 1; adj[c][b] = 1
        dst[i] = d; dst[j] = b
        done += 1
    assert done >= target, f"only {done}/{target} swaps accepted; raise the candidate batch"
    return [(nodes[u], nodes[v]) for u, v in zip(src, dst)], done


def null_a_for_period(G, blocs, rng, n_reps=N_REPS, num_trials=NUM_TRIALS):
    w = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    sav, nm = [], []
    for _ in range(n_reps):
        edges, _ = rewire_degree_preserving(G, rng=rng)
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        wp = rng.permutation(w)
        H.add_weighted_edges_from((u, v, float(wp[k])) for k, (u, v) in enumerate(edges))
        part, L, L1 = run_infomap(H, num_trials=num_trials, seed=int(rng.integers(1e9)))
        sav.append(1 - L / L1)
        nm.append(score_vs_blocs(part, blocs)[0])
    return np.array(sav), np.array(nm)


def null_b_2015(flows, blocs, n_reps=N_REPS, num_trials=NUM_TRIALS, seed=11):
    """Null B: keep the real 2015 corridor topology, permute the edge
    weights. Tests whether communities come merely from WHICH corridors
    exist rather than HOW MUCH flows along them."""
    sub = flows[flows.year0 == 2015]
    nodes = sorted(set(sub.orig) | set(sub.dest))
    edges = sub[sub[EST] > 0][["orig", "dest", EST]].values
    w = edges[:, 2].astype(float).copy()
    rng = np.random.default_rng(seed)
    sav, nm = [], []
    for _ in range(n_reps):
        wp = rng.permutation(w)
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(
            (e[0], e[1], wp[i]) for i, e in enumerate(edges))
        p, L, L1 = run_infomap(G, num_trials=num_trials,
                               seed=int(rng.integers(1e9)))
        sav.append(1 - L / L1)
        nm.append(score_vs_blocs(p, blocs)[0])
    sav, nm = np.array(sav), np.array(nm)
    np.save(f"{OUT}/results/nullB_savings_2015.npy", sav)
    np.save(f"{OUT}/results/nullB_nmi_2015.npy", nm)
    print(f"Null B (2015, {n_reps}x, {num_trials} trials): "
          f"savings {sav.mean():.4f}±{sav.std():.4f} max {sav.max():.4f} | "
          f"NMI {nm.mean():.4f} max {nm.max():.4f}")


def emp_p(null, observed):
    return (np.sum(null >= observed) + 1) / (len(null) + 1)


def main():
    flows = load_flows()
    graphs = build_graphs(flows)
    blocs = bloc_labels()
    obs = pd.read_csv(f"{OUT}/results/communities_metrics.csv").set_index("period")
    rng = np.random.default_rng(SEED)
    rows = []
    for y in PERIODS:
        sav, nm = null_a_for_period(graphs[y], blocs, rng)
        np.save(f"{OUT}/results/nullA_savings_{y}.npy", sav)
        np.save(f"{OUT}/results/nullA_nmi_{y}.npy", nm)
        o_sav, o_nmi = obs.loc[y, "compression_savings"], obs.loc[y, "infomap_nmi"]
        rows.append(dict(period=y,
                         obs_savings=o_sav, null_savings_mean=sav.mean(),
                         null_savings_sd=sav.std(), null_savings_max=sav.max(),
                         z_savings=(o_sav - sav.mean()) / sav.std(),
                         p_savings=emp_p(sav, o_sav),
                         excess_savings=o_sav - sav.mean(),
                         obs_nmi=o_nmi, null_nmi_mean=nm.mean(),
                         null_nmi_sd=nm.std(), null_nmi_max=nm.max(),
                         z_nmi=(o_nmi - nm.mean()) / nm.std(),
                         p_nmi=emp_p(nm, o_nmi)))
        print(f"{y}: savings obs {o_sav:.3f} vs null {sav.mean():.3f}±{sav.std():.3f} "
              f"(max {sav.max():.3f}, p={rows[-1]['p_savings']:.3f}, excess {rows[-1]['excess_savings']:.3f}) | "
              f"NMI obs {o_nmi:.3f} vs null {nm.mean():.3f}±{nm.std():.3f} "
              f"(max {nm.max():.3f}, p={rows[-1]['p_nmi']:.3f})", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/results/null_model.csv", index=False)
    null_b_2015(flows, blocs)


if __name__ == "__main__":
    main()
