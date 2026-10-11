"""All report figures. Consistent style, saved to out/figures/.

F1 total flow over time with the 1990s artifact annotated (milestone fix)
F2 alluvial: community membership across six periods
F3 choropleth: 2015 communities on a world map, legend keyed to the
   same top-destination tags as F2
F4 slope chart: PageRank rank shifts 1990 -> 2015 (blue/orange, not
   red/green - about 1 in 12 men cannot tell those apart)
F5 null models: observed 2015 structure vs both null distributions, plus
   observed vs Null A savings in every period
F6 gravity: predicted vs actual (all corridors) + top underpredicted
   corridors, with the Cambodia artifact bars hatched and grayed out
F7 conflict: outflow ratio, war vs peace origins
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from common import load_flows, PERIODS, OUT, DATA, EST, SUCCESSION

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
FIG = f"{OUT}/figures"


# ---------- cross-period community matching (stable ids and colors) ----------
def load_partitions():
    with open(f"{OUT}/results/partitions.json", encoding="utf-8") as fh:
        parts = json.load(fh)
    return {int(y): parts[str(y)]["infomap"] for y in parts}


def match_communities(parts):
    """Greedy Jaccard matching of modules across consecutive periods.
    Returns per-period dict module_id -> global_id. Countries chained through
    successor states."""
    gid_map = {}
    next_gid = 0
    prev_groups = None
    for y in PERIODS:
        part = parts[y]
        groups = {}
        for c, m in part.items():
            groups.setdefault(m, set()).add(SUCCESSION.get(c, c))
        gmap = {}
        if prev_groups is None:
            for m in sorted(groups, key=lambda k: -len(groups[k])):
                gmap[m] = next_gid; next_gid += 1
        else:
            pairs = []
            for m, s in groups.items():
                for pg, ps in prev_groups.items():
                    j = len(s & ps) / len(s | ps)
                    if j > 0:
                        pairs.append((j, m, pg))
            pairs.sort(reverse=True)
            used_m, used_g = set(), set()
            for j, m, pg in pairs:
                if m in used_m or pg in used_g or j < 0.2:
                    continue
                gmap[m] = pg; used_m.add(m); used_g.add(pg)
            for m in groups:
                if m not in gmap:
                    gmap[m] = next_gid; next_gid += 1
        gid_map[y] = gmap
        prev_groups = {gmap[m]: s for m, s in groups.items()}
    return gid_map


def gid_colors(parts, gid_map):
    """Stable colors: every community present in 2015 gets a tab20 color
    (largest first), remaining historical gids by size, rest gray."""
    sizes_15, sizes_all = {}, {}
    for y in PERIODS:
        for c, m in parts[y].items():
            g = gid_map[y][m]
            sizes_all[g] = sizes_all.get(g, 0) + 1
            if y == 2015:
                sizes_15[g] = sizes_15.get(g, 0) + 1
    order = sorted(sizes_15, key=lambda g: -sizes_15[g]) + \
        [g for g in sorted(sizes_all, key=lambda g: -sizes_all[g])
         if g not in sizes_15]
    cmap = plt.get_cmap("tab20")
    slots = [0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19]  # no grays
    colors = {g: cmap(slots[i]) for i, g in enumerate(order[: len(slots)])}
    return lambda g: colors.get(g, (0.90, 0.90, 0.90, 1.0))


def band_tags(parts, gid_map, year=2015, min_size=1):
    """global id -> 'USA+CAN' style tag: the two members with the highest
    in-strength in that year. F2 labels only bands with >= 6 members; F3's
    legend tags all of them."""
    h = pd.read_csv(f"{OUT}/results/hub_metrics.csv")
    ins = h[h.period == year].set_index("country").in_strength
    band = {}
    for c, m in parts[year].items():
        band.setdefault(gid_map[year][m], []).append(c)
    return {g: "+".join(sorted(mem, key=lambda c: -ins.get(c, 0))[:2])
            for g, mem in band.items() if len(mem) >= min_size}, \
           {g: len(mem) for g, mem in band.items()}


#  Figure 1: trend
def fig1():
    f = load_flows()
    tot = f.groupby("year0")[EST].sum() / 1e6
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = [f"{y}-\n{y+5}" for y in tot.index]
    ax.plot(range(len(tot)), tot.values, "o-", lw=2, color="#1f77b4", zorder=2)
    ax.scatter([1], [tot.iloc[1]], s=90, color="#d62728", zorder=3)
    for i, v in enumerate(tot.values):
        ax.annotate(f"{v:.0f}M", (i, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="0.35")
    pct_dip = 100 * (tot.iloc[1] / tot.iloc[0] - 1)
    pct_rise = 100 * (tot.iloc[-1] / tot.iloc[1] - 1)
    ax.annotate("1990-95 is inflated by one-off events: USSR &\n"
                "Yugoslavia dissolutions reclassified internal moves\n"
                "as international ones, plus Gulf War, Rwanda and\n"
                f"Balkan wars. The {pct_dip:.0f}% 'dip' to 1995-2000 is a\n"
                "return to baseline, not people migrating less;\n"
                f"flows then rise {pct_rise:.0f}% through 2015-2020.",
                xy=(1.08, tot.iloc[1]), xytext=(2.45, 69.2),
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4"),
                bbox=dict(boxstyle="round", fc="#fff5e0", ec="0.6"))
    ax.set_ylabel("Total estimated flow (millions / 5 yr)")
    ax.set_xticks(range(len(tot)), x)
    ax.set_title("Total estimated migration flow by five-year period")
    fig.tight_layout()
    fig.savefig(f"{FIG}/f1_total_flow.png")
    plt.close(fig)


# Figure 2: alluvial
def fig2(parts, gid_map, color_of):
    fig, ax = plt.subplots(figsize=(11, 7))
    xw, gap = 0.16, 2.0
    positions = {}
    for xi, y in enumerate(PERIODS):
        part = parts[y]
        groups = {}
        for c, m in part.items():
            groups.setdefault(gid_map[y][m], []).append(SUCCESSION.get(c, c))
        order = sorted(groups, key=lambda g: -len(groups[g]))
        y0 = 0.0
        for g in order:
            members = sorted(groups[g])
            h = len(members)
            ax.add_patch(plt.Rectangle((xi - xw / 2, y0), xw, h,
                                       color=color_of(g), ec="white", lw=0.4))
            for k, c in enumerate(members):
                positions[(y, c)] = (xi, y0 + k + 0.5, g)
            y0 += h + gap
    # ribbons
    for xi in range(len(PERIODS) - 1):
        y_a, y_b = PERIODS[xi], PERIODS[xi + 1]
        cs = set(c for (yy, c) in positions if yy == y_a) & \
             set(c for (yy, c) in positions if yy == y_b)
        # sorted: translucent ribbons composite in draw order, and set order
        # changes between python processes, so the PNG was never bit-identical
        for c in sorted(cs):
            xa, ya, g = positions[(y_a, c)]
            xb, yb, _ = positions[(y_b, c)]
            verts = [(xa + xw / 2, ya), (xa + 0.5, ya), (xb - 0.5, yb),
                     (xb - xw / 2, yb)]
            path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                                   MplPath.CURVE4, MplPath.CURVE4])
            ax.add_patch(PathPatch(path, fill=False, ec=color_of(g),
                                   lw=0.5, alpha=0.35))
    # label 2015 bands with their two highest-in-strength members
    tags, sizes = band_tags(parts, gid_map, min_size=6)
    band = {}
    for c, m in parts[2015].items():
        band.setdefault(gid_map[2015][m], []).append(c)
    xi = len(PERIODS) - 1
    for g, tag in tags.items():
        ys = [positions[(2015, SUCCESSION.get(c, c))][1] for c in band[g]
              if (2015, SUCCESSION.get(c, c)) in positions]
        ax.text(xi + xw / 2 + 0.05, np.mean(ys), tag,
                va="center", fontsize=8, color="0.2")
    ax.set_xlim(-0.5, len(PERIODS) + 0.35)
    ax.set_ylim(-2, 260)
    ax.set_xticks(range(len(PERIODS)), [f"{y}-{y+5}" for y in PERIODS])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Migration communities across periods (Infomap)\n"
                 "bands = communities, lines = one country's path; "
                 "2015 bands tagged by top destinations", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f2_alluvial.png")
    plt.close(fig)


# Figure 3 choropleth
def fig3(parts, gid_map, color_of):
    with open(f"{DATA}/ne110.geojson", encoding="utf-8") as fh:
        gj = json.load(fh)
    part = parts[2015]
    country_gid = {c: gid_map[2015][m] for c, m in part.items()}
    # 58 of the 230 network countries are too small for the 110m basemap
    # (Caribbean, Pacific, Gulf micro-states...). Caption says so.
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for feat in gj["features"]:
        code = feat["properties"]["ADM0_A3"]
        ne_alias = {"SDS": "SSD", "PSX": "PSE", "SAH": "ESH"}
        gid = country_gid.get(ne_alias.get(code, code))
        col = color_of(gid) if gid is not None else (0.92, 0.92, 0.92, 1)
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" \
            else [geom["coordinates"]]
        for poly in polys:
            for ring in poly[:1]:
                xs, ys = zip(*ring)
                ax.fill(xs, ys, color=col, ec="white", lw=0.3)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.axis("off")
    ax.set_title("Migration communities, 2015-2020 (Infomap on directed flows)")
    # legend: same tags as the F2 bands, largest community first
    from matplotlib.patches import Patch
    tags, sizes = band_tags(parts, gid_map)
    order = sorted(tags, key=lambda g: -sizes[g])
    handles = [Patch(facecolor=color_of(g), edgecolor="white",
                     label=f"{tags[g]} ({sizes[g]})") for g in order]
    handles.append(Patch(facecolor=(0.92, 0.92, 0.92, 1), edgecolor="white",
                         label="not in the flow data (Somaliland, N. Cyprus, Kosovo)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
              ncol=5, fontsize=7.5, frameon=False,
              title="community, tagged by its two largest destinations (size)",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f3_choropleth_2015.png", bbox_inches="tight")
    plt.close(fig)


# Figure 4 slope chart
def fig4():
    h = pd.read_csv(f"{OUT}/results/hub_metrics.csv")
    a = h[h.period == 1990].set_index("lineage").pagerank.rank(ascending=False)
    b = h[h.period == 2015].set_index("lineage").pagerank.rank(ascending=False)
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    show = sorted(set(a.nsmallest(15).index) | set(b.nsmallest(15).index))
    CAP = 26
    fig, ax = plt.subplots(figsize=(6.4, 8.2))

    def dodge(vals, min_gap=0.9):
        """Nudge overlapping label positions apart, preserving order."""
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = list(vals)
        for k in range(1, len(order)):
            i, j = order[k], order[k - 1]
            if out[i] - out[j] < min_gap:
                out[i] = out[j] + min_gap
        return out

    r0s = [min(a[c], CAP) for c in show]
    r1s = [min(b[c], CAP) for c in show]
    y0s, y1s = dodge(r0s), dodge(r1s)
    for c, r0v, r1v, y0, y1 in zip(show, r0s, r1s, y0s, y1s):
        true0, true1 = int(a[c]), int(b[c])
        col = "#1f77b4" if true1 < true0 else ("#ff7f0e" if true1 > true0 else "0.5")
        ls = "--" if (a[c] > CAP or b[c] > CAP) else "-"
        ax.plot([0, 1], [r0v, r1v], ls, color=col, lw=1.4, alpha=0.8)
        l0 = f"{c} {true0}" if a[c] <= CAP else f"{c} ({true0})"
        l1 = f"{true1} {c}" if b[c] <= CAP else f"({true1}) {c}"
        ax.text(-0.02, y0, l0, ha="right", va="center", fontsize=8)
        ax.text(1.02, y1, l1, ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.25, 1.25)
    # the beyond-CAP labels get dodged below CAP; give them room instead of
    # letting them run into the x tick labels
    bottom = max(max(y0s), max(y1s)) + 1.0
    ax.set_ylim(bottom + 0.6, 0)
    ax.text(0.5, bottom + 0.3, "dashed = beyond rank 26 (true rank in parentheses)",
            ha="center", fontsize=7.5, color="0.4")
    ax.set_xticks([0, 1], ["1990-1995", "2015-2020"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("PageRank rank, 1990 vs 2015\n(blue = rising hub, orange = falling)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/f4_slope_pagerank.png")
    plt.close(fig)


# Figure 5: null models
def fig5():
    obs = pd.read_csv(f"{OUT}/results/communities_metrics.csv").set_index("period")
    nul = pd.read_csv(f"{OUT}/results/null_model.csv").set_index("period")
    a_sav = np.load(f"{OUT}/results/nullA_savings_2015.npy")
    a_nmi = np.load(f"{OUT}/results/nullA_nmi_2015.npy")
    b_sav = np.load(f"{OUT}/results/nullB_savings_2015.npy")
    b_nmi = np.load(f"{OUT}/results/nullB_nmi_2015.npy")
    CA, CB, CO = "#fdae6b", "#9ecae1", "#d62728"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 0.95]})
    for ax, a, b, obs_val, lab in [
            (axes[0], a_sav, b_sav, obs.loc[2015, "compression_savings"],
             "Compression savings (map equation)"),
            (axes[1], a_nmi, b_nmi, obs.loc[2015, "infomap_nmi"],
             "NMI vs real-world blocs")]:
        bins = np.linspace(0, 1.06 * max(obs_val, a.max(), b.max()), 36)
        ax.hist(a, bins=bins, color=CA, alpha=0.9,
                label="Null A: degrees kept, wiring\nrewired, weights shuffled (100x)")
        ax.hist(b, bins=bins, color=CB, alpha=0.75,
                label="Null B: real corridors,\nweights shuffled (100x)")
        ax.axvline(obs_val, color=CO, lw=2, label="Observed 2015")
        ax.set_xlabel(lab); ax.set_ylabel("Null replicates")
        ax.set_ylim(0, 1.55 * max(np.histogram(a, bins)[0].max(),
                                  np.histogram(b, bins)[0].max()))  # room for the legend
        ax.legend(fontsize=7.5, loc="upper center", frameon=False)
    # panel 3: does the decline in savings survive the null? observed vs Null A, all periods
    ax = axes[2]
    x = range(len(PERIODS))
    ax.fill_between(x, nul.null_savings_mean - 2 * nul.null_savings_sd,
                    nul.null_savings_mean + 2 * nul.null_savings_sd,
                    color=CA, alpha=0.45, label="Null A mean ± 2 sd")
    ax.plot(x, nul.null_savings_mean, "s-", color="#e6550d", ms=4, lw=1.2)
    ax.plot(x, obs.loc[PERIODS, "compression_savings"], "o-", color=CO, lw=2,
            label="observed")
    for xi, (o, n) in enumerate(zip(obs.loc[PERIODS, "compression_savings"],
                                    nul.null_savings_mean)):
        ax.annotate(f"+{o - n:.3f}", (xi, o), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="0.3")
    ax.set_xticks(list(x), [f"{y}" for y in PERIODS], fontsize=8)
    ax.set_ylim(0, None)
    ax.set_ylabel("Compression savings")
    ax.set_title("All periods: observed vs Null A\n(labels = excess over null mean)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    fig.suptitle("Observed community structure vs null models")
    fig.tight_layout()
    fig.savefig(f"{FIG}/f5_null_model.png")
    plt.close(fig)


# Figure 6: gravity
def fig6():
    res = pd.read_csv(f"{OUT}/results/gravity_residuals.csv.gz")
    fit = pd.read_csv(f"{OUT}/results/gravity_fit.csv", index_col=0).iloc[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    hb = ax.hexbin(np.log1p(res.pred), np.log1p(res[EST]), gridsize=45,
                   bins="log", cmap="Blues")
    lim = [0, max(np.log1p(res[EST]).max(), np.log1p(res.pred).max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("ln(1 + predicted flow)  [PPML gravity]")
    ax.set_ylabel("ln(1 + actual flow)")
    ax.set_title(f"Gravity fit (all corridors)\n"
                 f"held-out Spearman {fit.test_spearman_full:.2f} "
                 f"vs {fit.test_spearman_base:.2f} distance-only")
    fig.colorbar(hb, ax=ax, label="corridors (log count)")

    ax = axes[1]
    top = pd.read_csv(f"{OUT}/results/gravity_top_underpredicted.csv").head(12)
    labels = [f"{o}\u2192{d} '{str(y)[2:]}" for y, o, d in
              zip(top.year0, top.orig, top.dest)]
    # the X->KHM 1990-95 corridors are the estimation artifact discussed in
    # P2; they belong in the chart (the model does miss them) but must not be
    # read as displacement, so they get their own look
    artifact = (top.dest == "KHM").values
    ys = np.array(range(len(top))[::-1])
    ax.barh(ys[~artifact], top.ratio[~artifact], color="#d62728", alpha=0.85,
            label="conflict, displacement, return migration")
    ax.barh(ys[artifact], top.ratio[artifact], color="0.75", edgecolor="0.4",
            hatch="///", label="Cambodia 1990-95 estimation artifact")
    ax.set_yticks(ys, labels, fontsize=8)
    ax.set_xlabel("actual / predicted flow")
    ax.set_title("Corridors most under-predicted by economics\n"
                 "(flows >= 50k; hatched = known data artifact, not migration)")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f6_gravity.png")
    plt.close(fig)


# Figure 7: conflict
def fig7():
    m = pd.read_csv(f"{OUT}/results/origin_period_residuals_war.csv")
    t = pd.read_csv(f"{OUT}/results/conflict_test.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    data = [m[~m.war].outflow_log_ratio.dropna(), m[m.war].outflow_log_ratio.dropna()]
    bp = ax.boxplot(data, tick_labels=["no war\nat origin", "war at origin\n(UCDP intensity 2)"],
                    showfliers=False, patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], ["#9ecae1", "#fc9272"]):
        patch.set_facecolor(col)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(rng.normal(i + 1, 0.06, len(d)), d, s=4, alpha=0.25, color="0.3")
    ax.axhline(0, color="0.5", ls=":")
    p = t.loc["outflow_log_ratio", "p_value"]
    ax.set_ylabel("ln(total outflow / gravity-predicted outflow)")
    ax.set_title(f"War origins emigrate beyond economic prediction\n"
                 f"Mann-Whitney one-sided p = {p:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/f7_conflict.png")
    plt.close(fig)


if __name__ == "__main__":
    parts = load_partitions()
    gm = match_communities(parts)
    color_of = gid_colors(parts, gm)
    fig1(); fig2(parts, gm, color_of); fig3(parts, gm, color_of)
    fig4(); fig5(); fig6(); fig7()
    print("figures written to", FIG)
