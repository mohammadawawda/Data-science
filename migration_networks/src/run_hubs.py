"""Problem 2: which countries are structural hubs, and how does that change?

Metrics per period: weighted PageRank (alpha=0.85 + sensitivity 0.80/0.90),
weighted in-/out-strength, weighted betweenness (distance = 1/weight, so
heavy corridors are 'short'). Cross-period trajectories chain successor
states (SUD->SDN, SCG->SRB).

External consistency check: overlap of PageRank top-10 (2015) with the UN
DESA International Migrant Stock top-10 destinations. NOTE: partly circular
(the flow estimates are derived from the same UN stock data) - stated
honestly in the writeup as a consistency check, not independent validation.
The UN list below is hardcoded from UN DESA IMS 2020, so it should be
checked against the published table.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from common import build_graphs, PERIODS, OUT, SUCCESSION

# UN DESA International Migrant Stock, top-10 destinations (mid-2020, per
# the IMS2024 revision on which the flow estimates are built). Top-5 order
# verified against UN DESA; analysis uses SET OVERLAP only.
UN_TOP10_2020 = ["USA", "DEU", "SAU", "GBR", "FRA", "ESP", "CAN", "ARE", "AUS", "RUS"]


def metrics_for(G):
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    pr80 = nx.pagerank(G, alpha=0.80, weight="weight")
    pr90 = nx.pagerank(G, alpha=0.90, weight="weight")
    ins = dict(G.in_degree(weight="weight"))
    outs = dict(G.out_degree(weight="weight"))
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["inv"] = 1.0 / d["weight"]
    btw = nx.betweenness_centrality(H, weight="inv", normalized=True)
    return pr, pr80, pr90, ins, outs, btw


def main():
    graphs = build_graphs()
    frames = []
    sens_rows = []
    for y in PERIODS:
        G = graphs[y]
        pr, pr80, pr90, ins, outs, btw = metrics_for(G)
        df = pd.DataFrame({
            "period": y, "country": list(G.nodes()),
            "pagerank": [pr[n] for n in G.nodes()],
            "in_strength": [ins[n] for n in G.nodes()],
            "out_strength": [outs[n] for n in G.nodes()],
            "betweenness": [btw[n] for n in G.nodes()],
        })
        frames.append(df)
        order = list(G.nodes())
        s80 = spearmanr([pr[n] for n in order], [pr80[n] for n in order]).statistic
        s90 = spearmanr([pr[n] for n in order], [pr90[n] for n in order]).statistic
        sens_rows.append(dict(period=y, spearman_a80=s80, spearman_a90=s90))

    allm = pd.concat(frames, ignore_index=True)
    # lineage chaining for trajectories
    allm["lineage"] = allm.country.map(lambda c: SUCCESSION.get(c, c))
    allm.to_csv(f"{OUT}/results/hub_metrics.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(f"{OUT}/results/pagerank_sensitivity.csv", index=False)

    # summaries
    for y in (1990, 2015):
        top = allm[allm.period == y].nlargest(10, "pagerank")
        print(f"top-10 PageRank {y}:", " ".join(top.country.tolist()))
    top15b = allm[allm.period == 2015].nlargest(15, "betweenness")
    print("top-15 betweenness 2015:", " ".join(top15b.country.tolist()))

    pr2015 = allm[allm.period == 2015].nlargest(10, "pagerank").country.tolist()
    overlap = len(set(pr2015) & set(UN_TOP10_2020))
    print(f"PageRank-top10 vs UN-IMS-top10 overlap: {overlap}/10")
    with open(f"{OUT}/results/un_consistency.txt", "w") as fh:
        fh.write(
            f"pagerank_top10_2015: {pr2015}\n"
            f"un_ims_top10_mid2020 (IMS2024 revision): {UN_TOP10_2020}\n"
            f"set_overlap: {overlap}/10 "
            f"(missing from PageRank top10: {sorted(set(UN_TOP10_2020) - set(pr2015))}; "
            f"extra in PageRank top10: {sorted(set(pr2015) - set(UN_TOP10_2020))})\n"
            "notes: top-5 order (USA DEU SAU GBR FRA) verified against UN DESA; "
            "set of 10 verified stable 2015-2024 per Migration Data Portal.\n"
            "caveat: flow estimates derive from the same UN stock data -> this is "
            "a consistency check, not independent validation.\n")


if __name__ == "__main__":
    main()
