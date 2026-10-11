"""Problem 1: community detection on each period's directed weighted graph.

Methods: Infomap (directed flow model, primary) and Louvain (directed
modularity, comparison). Validation: NMI/ARI of detected partitions against
ground-truth free-movement blocs, vs two baselines (continent partition,
random permutations of the detected partition).
"""
import json
import numpy as np
import pandas as pd
import networkx as nx
from infomap import Infomap
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from common import build_graphs, bloc_labels, continent_labels, PERIODS, OUT

SEED = 42


def run_infomap(G, num_trials=20, seed=SEED):
    """Returns (partition dict node->module, codelength, one_level_codelength)."""
    idx = {n: i for i, n in enumerate(G.nodes())}
    rev = {i: n for n, i in idx.items()}
    im = Infomap(directed=True, num_trials=num_trials, seed=seed, silent=True)
    for u, v, d in G.edges(data=True):
        im.add_link(idx[u], idx[v], d["weight"])
    im.run()
    part = {rev[nid]: mod for nid, mod in im.get_modules().items()}
    return part, im.codelength, im.one_level_codelength


def run_louvain(G, seed=SEED):
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    part = {n: i for i, c in enumerate(comms) for n in c}
    Q = nx.community.modularity(G, comms, weight="weight")
    return part, Q


def score_vs_blocs(partition, blocs):
    """NMI/ARI restricted to countries that belong to a ground-truth bloc
    and appear in the partition."""
    common = [c for c in blocs if c in partition]
    y_true = [blocs[c] for c in common]
    y_pred = [partition[c] for c in common]
    return nmi(y_true, y_pred), ari(y_true, y_pred), len(common)


def random_baseline(partition, blocs, n=1000, seed=SEED):
    """Permute the detected labels among the same nodes -> distribution of NMI
    achievable by a partition with identical group sizes but no relation to
    geography."""
    rng = np.random.default_rng(seed)
    common = [c for c in blocs if c in partition]
    y_true = [blocs[c] for c in common]
    labels = np.array([partition[c] for c in common])
    vals = []
    for _ in range(n):
        vals.append(nmi(y_true, rng.permutation(labels)))
    return float(np.mean(vals)), float(np.percentile(vals, 95))


def main():
    graphs = build_graphs()
    blocs = bloc_labels()
    rows, parts_store = [], {}
    for y in PERIODS:
        G = graphs[y]
        cont = continent_labels(list(G.nodes()))

        im_part, L, L1 = run_infomap(G)
        lv_part, Q = run_louvain(G)

        im_nmi, im_ari, ncommon = score_vs_blocs(im_part, blocs)
        lv_nmi, lv_ari, _ = score_vs_blocs(lv_part, blocs)
        ct_nmi, ct_ari, _ = score_vs_blocs({c: v for c, v in cont.items()}, blocs)
        rnd_mean, rnd_p95 = random_baseline(im_part, blocs)

        rows.append(dict(period=y,
                         infomap_modules=len(set(im_part.values())),
                         louvain_modules=len(set(lv_part.values())),
                         codelength=L, one_level_codelength=L1,
                         compression_savings=1 - L / L1,
                         louvain_modularity=Q,
                         infomap_nmi=im_nmi, infomap_ari=im_ari,
                         louvain_nmi=lv_nmi, louvain_ari=lv_ari,
                         continent_nmi=ct_nmi, continent_ari=ct_ari,
                         random_nmi_mean=rnd_mean, random_nmi_p95=rnd_p95,
                         n_bloc_countries=ncommon))
        parts_store[y] = {"infomap": im_part, "louvain": lv_part}
        print(f"{y}: infomap {len(set(im_part.values()))} modules, "
              f"savings {1-L/L1:.3f}, NMI {im_nmi:.3f} "
              f"(continent {ct_nmi:.3f}, random {rnd_mean:.3f}) | "
              f"louvain {len(set(lv_part.values()))} modules Q={Q:.3f} NMI {lv_nmi:.3f}")

    pd.DataFrame(rows).to_csv(f"{OUT}/results/communities_metrics.csv", index=False)
    with open(f"{OUT}/results/partitions.json", "w") as fh:
        json.dump(parts_store, fh)
    # readable membership table for 2015
    inv = {}
    for c, m in parts_store[2015]["infomap"].items():
        inv.setdefault(m, []).append(c)
    with open(f"{OUT}/results/communities_2015_readable.txt", "w") as fh:
        for m in sorted(inv, key=lambda k: -len(inv[k])):
            fh.write(f"module {m} ({len(inv[m])}): {' '.join(sorted(inv[m]))}\n\n")


if __name__ == "__main__":
    main()
