"""Sensitivity check: do the hub findings survive a different estimation
method?

We rebuild the 1990 and 2015 graphs using sd_drop_neg (stock differencing,
an estimator with completely different mechanics from demographic
accounting) and compare PageRank rankings with the primary da_pb_closed
results. Two questions:
  1. Are the overall rankings stable? (rank correlation on common countries)
  2. Does Cambodia's 1990 top-5 position survive? If it collapses under the
     alternative estimator, that supports reading it as a
     demographic-accounting artifact rather than a real hub.
"""
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from common import load_flows, PERIODS, OUT

ALT = "sd_drop_neg"


def graph_for(flows, y, col):
    sub = flows[(flows.year0 == y) & (flows[col] > 0)]
    G = nx.DiGraph()
    G.add_nodes_from(sorted(set(flows[flows.year0 == y].orig)))
    G.add_weighted_edges_from(sub[["orig", "dest", col]].itertuples(index=False, name=None))
    return G


def main():
    flows = load_flows()
    primary = pd.read_csv(f"{OUT}/results/hub_metrics.csv")
    rows = []
    for y in (1990, 2015):
        G = graph_for(flows, y, ALT)
        pr_alt = nx.pagerank(G, alpha=0.85, weight="weight")
        p = primary[primary.period == y].set_index("country").pagerank
        common = [c for c in p.index if c in pr_alt]
        rho = spearmanr(p[common], [pr_alt[c] for c in common]).statistic
        rank_alt = pd.Series(pr_alt).rank(ascending=False)
        rank_pri = p.rank(ascending=False)
        top10_alt = pd.Series(pr_alt).nlargest(10).index.tolist()
        top10_pri = p.nlargest(10).index.tolist()
        rows.append(dict(period=y, spearman_all=rho,
                         top10_overlap=len(set(top10_alt) & set(top10_pri)),
                         khm_rank_primary=int(rank_pri.get("KHM", -1)),
                         khm_rank_alt=int(rank_alt.get("KHM", -1))))
        print(f"{y}: Spearman(da_pb_closed, {ALT}) = {rho:.3f}; "
              f"top-10 overlap {rows[-1]['top10_overlap']}/10; "
              f"KHM rank {rows[-1]['khm_rank_primary']} -> {rows[-1]['khm_rank_alt']} under {ALT}")
    pd.DataFrame(rows).to_csv(f"{OUT}/results/sensitivity_estimator.csv", index=False)


if __name__ == "__main__":
    main()
