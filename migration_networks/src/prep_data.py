"""Step 0: data preparation. Reproduces every derived file in data/ from the
raw downloads. Safe to re-run; steps are skipped if their raw input is
absent (the repo ships the small derived files, so analyses run without the
1.25 GB CEPII archive).

Raw inputs expected (see data/PROVENANCE.txt for exact versions):
  uploads/bilat_mig.csv            Abel & Cohen flow estimates, Version 8
  uploads/Gravity_csv_V202211.zip  CEPII Gravity database V202211
  uploads/ucdp-prio-acd-261-csv.zip UCDP/PRIO Armed Conflict v26.1
Derived outputs:
  data/flows.csv           verbatim copy of the flow CSV
  data/gravity_subset.csv.gz   CEPII filtered to 6 years x 20 columns
  data/gravity_dedup.csv.gz    subset deduplicated per (year, iso3_o, iso3_d)
  data/UcdpPrioConflict_v26_1.csv
  data/ne110.geojson       Natural Earth 110m admin-0 (GitHub mirror)
"""
import os, shutil, zipfile, urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = str(PROJECT_ROOT / "data")
UP = os.environ.get("MIGRATION_RAW_DATA_DIR", str(PROJECT_ROOT / "uploads"))

KEEP = ["year", "iso3_o", "iso3_d", "dist", "distcap", "contig",
        "comlang_off", "comlang_ethno", "comcol", "col_dep_ever", "col45",
        "comrelig", "pop_o", "pop_d", "gdp_o", "gdp_d", "gdpcap_o",
        "gdpcap_d", "eu_o", "eu_d"]
YEARS = {1990, 1995, 2000, 2005, 2010, 2015, 2019}
NEED = ["dist", "gdp_o", "gdp_d", "pop_o", "pop_d",
        "contig", "comlang_off", "col_dep_ever", "comcol"]
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          "master/geojson/ne_110m_admin_0_countries.geojson")


def prep_flows():
    for cand in ["bilat_mig.csv", "bilat_mig__1_.csv", "bilat_mig (1).csv"]:
        src = os.path.join(UP, cand)
        if os.path.exists(src):
            shutil.copy(src, f"{DATA}/flows.csv")
            n = sum(1 for _ in open(f"{DATA}/flows.csv")) - 1
            assert n == 313736, f"unexpected row count {n}; expected V8"
            print(f"flows.csv: {n:,} rows (Version 8 fingerprint OK)")
            return
    print("flows raw not found; keeping existing data/flows.csv")


def prep_gravity():
    src = os.path.join(UP, "Gravity_csv_V202211.zip")
    if not os.path.exists(src):
        print("CEPII zip not found; keeping existing gravity files")
        return
    z = zipfile.ZipFile(src)
    chunks = []
    with z.open("Gravity_V202211.csv") as fh:
        for ch in pd.read_csv(fh, usecols=KEEP, chunksize=500_000):
            chunks.append(ch[ch.year.isin(YEARS)])
    g = pd.concat(chunks, ignore_index=True)
    g.to_csv(f"{DATA}/gravity_subset.csv.gz", index=False, compression="gzip")
    # CEPII carries a row for every country_id sharing an ISO3 in every year;
    # the defunct entity's rows are all-NaN outside its lifespan. Keep the
    # most-populated row per dyad-year. Verified: no dyad-year has two
    # populated rows, so no corridor is double counted.
    g = g[g.year.isin(YEARS - {2019})].copy()
    g["nulls"] = g[NEED].isna().sum(axis=1)
    two_pop = (g[g.nulls == 0]
               .duplicated(["year", "iso3_o", "iso3_d"], keep=False).sum())
    assert two_pop == 0, "double-populated dyad-year found"
    g = (g.sort_values("nulls")
         .drop_duplicates(["year", "iso3_o", "iso3_d"], keep="first")
         .drop(columns="nulls"))
    assert not g.duplicated(["year", "iso3_o", "iso3_d"]).any()
    g.to_csv(f"{DATA}/gravity_dedup.csv.gz", index=False, compression="gzip")
    print(f"gravity: subset + dedup written ({len(g):,} unique dyad-years)")


def prep_ucdp():
    src = os.path.join(UP, "ucdp-prio-acd-261-csv.zip")
    if not os.path.exists(src):
        print("UCDP zip not found; keeping existing csv")
        return
    with zipfile.ZipFile(src) as z:
        z.extractall(DATA)
    print("UCDP extracted")


def prep_basemap():
    dst = f"{DATA}/ne110.geojson"
    if os.path.exists(dst):
        print("ne110.geojson present")
        return
    urllib.request.urlretrieve(NE_URL, dst)
    print("ne110.geojson downloaded")


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    prep_flows(); prep_gravity(); prep_ucdp(); prep_basemap()
