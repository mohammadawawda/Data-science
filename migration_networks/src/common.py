"""Shared foundation: load flows, build per-period directed graphs, ground-truth blocs.

Where the World Moves. Estimate column: da_pb_closed (pseudo-Bayesian
demographic accounting, closed system) - the best-performing method in Abel &
Cohen's validation against reported flow statistics.
"""
from pathlib import Path

import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = str(PROJECT_ROOT / "data")
OUT = str(PROJECT_ROOT / "out")
EST = "da_pb_closed"
PERIODS = [1990, 1995, 2000, 2005, 2010, 2015]

# Lineage chains for cross-period tracking (predecessor -> dominant successor)
SUCCESSION = {"SUD": "SDN", "SCG": "SRB"}


def load_flows():
    f = pd.read_csv(f"{DATA}/flows.csv")
    f = f[f.orig != f.dest].copy()  # self-loops are all zero anyway; be explicit
    return f


def build_graphs(flows=None):
    """dict period -> weighted DiGraph using only that period's countries,
    edges with positive estimated flow."""
    if flows is None:
        flows = load_flows()
    graphs = {}
    for y in PERIODS:
        sub = flows[(flows.year0 == y) & (flows[EST] > 0)]
        G = nx.DiGraph()
        nodes = sorted(set(flows[flows.year0 == y].orig) | set(flows[flows.year0 == y].dest))
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(sub[["orig", "dest", EST]].itertuples(index=False, name=None))
        graphs[y] = G
    return graphs


#  Ground-truth blocs (free-movement / integration areas) ----
# Memberships as of ~2015; used to validate detected communities.
BLOCS = {
    "EU": ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
           "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
           "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR"],
    "GCC": ["SAU", "ARE", "KWT", "QAT", "BHR", "OMN"],
    "ECOWAS": ["BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR",
               "MLI", "NER", "NGA", "SEN", "SLE", "TGO"],
    "MERCOSUR": ["ARG", "BRA", "PRY", "URY"],
    "POST_SOVIET": ["RUS", "UKR", "BLR", "MDA", "GEO", "ARM", "AZE", "KAZ",
                    "KGZ", "TJK", "TKM", "UZB"],
    "ASEAN": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"],
}


def bloc_labels():
    """country -> bloc name, for countries in exactly one bloc."""
    lab = {}
    for b, members in BLOCS.items():
        for c in members:
            assert c not in lab, f"{c} in two blocs"
            lab[c] = b
    return lab


#  Continent mapping (baseline partition) 
def continent_labels(codes):
    import pycountry_convert as pcc
    manual = {
        # entities pycountry-convert cannot resolve, assigned by geography
        "ESH": "AF", "PSE": "AS", "TWN": "AS", "XKX": "EU", "SCG": "EU",
        "SUD": "AF", "COK": "OC", "NIU": "OC", "TKL": "OC", "WLF": "OC",
        "GLP": "NA", "MTQ": "NA", "GUF": "SA", "REU": "AF", "MYT": "AF",
        "SPM": "NA", "SHN": "AF", "FLK": "SA", "AIA": "NA", "MSR": "NA",
        "VGB": "NA", "CUW": "NA", "SXM": "NA", "BMU": "NA", "GIB": "EU",
        "FRO": "EU", "GRL": "NA", "IMN": "EU", "PYF": "OC", "NCL": "OC",
        "GUM": "OC", "ASM": "OC", "MNP": "OC", "VIR": "NA", "PRI": "NA",
        "CYM": "NA", "TCA": "NA", "MAC": "AS", "HKG": "AS", "TLS": "AS",
    }
    out = {}
    for c in codes:
        if c in manual:
            out[c] = manual[c]
            continue
        try:
            a2 = pcc.country_alpha3_to_country_alpha2(c)
            out[c] = pcc.country_alpha2_to_continent_code(a2)
        except Exception:
            out[c] = None
    return out
