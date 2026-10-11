# Where the World Moves — Mapping International Migration Networks

Made by Mohammad Awawda

---

## 1. What this project does

We model 30 years of international migration (1990–2020) as six directed,
weighted networks, one per five-year period. Nodes are countries
(228–230 per period); an edge u→v is the estimated number of people who
moved from u to v in that period. On this network we tackle three problems:

Problem1- Community structure: Do migration flows organize the world into
coherent "migration systems"? We detect communities with Infomap
(a directed flow-based method) and Louvain (directed modularity),
validate them against real-world free-movement blocs (EU, GCC, ECOWAS,
Mercosur, post-Soviet space, ASEAN), and test against two null models
(degree-preserving rewiring; weight shuffling) to show the structure is
not a degree or density artifact. Headline findings: communities follow
colonial-linguistic-labor systems rather than continents (Spain sits with
Hispanic America; the Gulf with South Asia; Israel with the post-Soviet
space), alignment with formal blocs rises over time (NMI 0.69 → 0.78),
while overall modularity falls (compression savings 0.178 → 0.118, and
0.081 → 0.054 after subtracting what the rewiring null gives for free):
the network is de-regionalizing. On the bloc benchmark Louvain is as good
as Infomap (its ARI is higher in 5 of 6 periods); we keep Infomap for its
directed-flow reading and finer resolution, not for the score.

Problem 2- Hubs and their evolution: Which countries are structurally
central, and how did that change? Weighted PageRank (with damping sensitivity), in/out-strength, and weighted betweenness per period.
Findings: USA is rank 1 throughout; Russia, Ukraine, Spain, Turkey and Kazakhstan rise; Germany, France, Canada slip in rank; betweenness surfaces transit countries (India, Saudi Arabia, Bangladesh, Nigeria)
that volume rankings hide. Cambodia's spurious 1990 rank (4th) is used as
a case study of estimate artifacts , it collapses to rank 130 under an alternative estimator.

Problem 3- Economics, geography, and what they miss: A **PPML gravity
model** (distance, GDP, population, contiguity, language, colonial ties,
period fixed effects) explains corridor ordering well out-of-sample
(Spearman 0.62 vs 0.34 for distance alone), with textbook elasticities ,and a notable zero on origin GDP. The corridors the model misses worst
(200–800× underprediction) are uniformly conflict and displacement
corridors; formally, war origins (UCDP intensity-2) emigrate beyond
economic prediction (Mann-Whitney p = 0.012).

## 2. Datasets — which, for what, from where

| dataset | size | role in the project |
|---|---|---|
| **Abel & Cohen bilateral migration flow estimates**, figshare item 7731233, **Version 8** (IMS2024 + WPP2024 inputs) | 9.7 MB CSV, 313,736 rows × 9 cols | The network itself. Column `da_pb_closed` (pseudo-Bayesian demographic accounting — the authors' best-validated method) is the edge weight for every graph in P1–P3. |
| **CEPII Gravity V202211** | 207 MB zip → 1.25 GB CSV, 4,699,296 rows × 87 cols | Covariates for the P3 gravity model: distance, GDP, population, contiguity, common language, colonial ties. Filtered to 6 years × 20 cols and deduplicated (`data/gravity_dedup.csv.gz`, 354,294 dyad-years). |
| **UCDP/PRIO Armed Conflict v26.1** | 0.52 MB CSV, 2,816 rows × 28 cols | War flags for P3's conflict analysis: a country-period is "war" if any located conflict-year at intensity 2 (≥1000 battle deaths) falls in its window. |
| **Natural Earth 110m admin-0** (GitHub mirror) | 0.8 MB geojson, 177 features | Basemap for the community choropleth (visualization only). |
| **UN DESA IMS top-10 destinations** (hardcoded, verified) | 10 codes | External consistency check for P2 hub rankings (set overlap only; semi-circular with the flow source, stated as such). |

Full provenance, version fingerprints and known data issues:
`data/PROVENANCE.txt`.

## 3. Repository layout — purpose of every file

```
src/                       all code (Python 3.12; see requirements.txt)
  common.py                shared foundation: flow loader, per-period DiGraph
                           builder, ground-truth bloc memberships, continent
                           mapping, succession chains (SUD→SDN, SCG→SRB)
  prep_data.py             STEP 0: derives every file in data/ from the raw
                           downloads (CEPII filter + dedup, UCDP extract,
                           basemap download); safe to re-run
  run_communities.py       STEP 1 (P1): Infomap (directed, 20 trials, seed
                           42) + Louvain per period; NMI/ARI vs blocs;
                           continent + random-permutation baselines
  run_null.py              STEP 2 (P1 validation): Null A = degree-
                           preserving directed edge swaps + permuted
                           weights (100/period, trial parity); Null B =
                           real topology with permuted weights (100×,
                           2015, trial parity). ~30 min.
  run_hubs.py              STEP 3 (P2): PageRank α=.85 (+ .80/.90
                           sensitivity), in/out-strength, betweenness
                           (dist = 1/weight); UN consistency check
  run_gravity.py           STEP 4 (P3): PPML gravity (statsmodels GLM
                           Poisson), 80/20 holdout vs distance-only
                           baseline, HC1 + dyad-clustered SEs, residual and
                           top-corridor extraction
  run_conflict.py          STEP 5 (P3): UCDP location→ISO3 mapping (full-
                           coverage assert), war flags with succession fix,
                           Mann-Whitney tests on three origin-level metrics
  run_sensitivity.py       STEP 6: alternative-estimator check
                           (sd_drop_neg): ranking stability + the Cambodia
                           artifact collapse
  run_figures.py           STEP 7: all seven report figures, incl. cross-
                           period community matching for stable colors;
                           F3 legend and F2 tags share one helper
  run_all.py               orchestrator: runs steps 0–7 in order (~30 min)

data/
  PROVENANCE.txt           datasets, exact versions, fingerprints, caveats
  flows.csv                Abel & Cohen V8 (copy of the raw download)
  gravity_dedup.csv.gz     CEPII deduplicated per dyad-year (derived)
  UcdpPrioConflict_v26_1.csv
  ne110.geojson            basemap
  Countries_V202211.csv    CEPII country list (code crosswalk reference)

out/results/               every number in the writeup lives here
  communities_metrics.csv  per-period module counts, codelengths, NMI/ARI
                           vs blocs + both baselines
  partitions.json          full Infomap + Louvain partitions, all periods
  communities_2015_readable.txt  human-readable 2015 module memberships
  null_model.csv           Null A per period: null mean/sd/max, z, empirical
                           p, excess savings (observed - null mean)
  nullA_*_{period}.npy, nullB_*_2015.npy   null distributions (savings, NMI)
  hub_metrics.csv          PageRank / strengths / betweenness, all periods
  pagerank_sensitivity.csv damping-parameter Spearman correlations
  un_consistency.txt       PageRank vs UN IMS top-10, with caveat
  sensitivity_estimator.csv  sd_drop_neg vs da_pb_closed rankings
  gravity_coefficients*.csv  PPML coefficients (HC1 and dyad-clustered SEs)
  gravity_fit.csv          holdout + pseudo-R² metrics, full vs baseline
  gravity_residuals.csv.gz corridor-level predictions and residuals
  gravity_top_underpredicted.csv / _overpredicted.csv
  war_flags.csv            country-period war flags (UCDP intensity 2)
  conflict_test.csv        Mann-Whitney results, three metrics
  origin_period_residuals_war.csv  per-origin aggregates behind fig. 7

out/figures/               f1..f7, the seven report figures
```

## 4. How to reproduce

```bash
# from the project root, the directory containing README.md
pip install -r requirements.txt
python src/run_all.py          # steps 0–7, ~30 minutes
```

The shipped `data/` directory contains the derived files needed by the
analysis pipeline, including `gravity_dedup.csv.gz`, so the project can run
without downloading the huge raw CEPII archive. If you want to rebuild the
derived data from raw downloads, place the raw files in `uploads/` at the
project root or set `MIGRATION_RAW_DATA_DIR` to that folder; when those raw files
are absent, `prep_data.py` keeps the included derived files.

All randomized steps are seeded (Infomap seed 42 / 20 trials; nulls seeds
7 and 11; holdout split seed 42): the full pipeline reproduces every
stored number exactly — we verified this with a clean-slate rerun diffed
against the archived outputs at 1e-9 tolerance.

## 5. Honesty notes (read before trusting anything)

The flows are model estimates derived from migrant-stock changes, not
observed moves. Known consequences we handle explicitly: the 1990-95
total is inflated by the USSR/Yugoslavia dissolutions (annotated in
fig. 1); the estimated 1990-95 USA→Cambodia flow (431k) exceeds the
Cambodian-born US population of the era and is treated as an estimation
artifact, supported by its collapse under an alternative estimator; the
pseudo-Bayesian method is calibrated mostly on intra-European flows and
may push totals toward the upper end. Every headline number in the
writeup was recomputed from the data in an audit pass (including one bug
we found and fixed: war flags for pre-2010 Sudan and Serbia-Montenegro
originally failed to merge due to successor-code mismatch).

Two more things we got wrong the first time and changed. Our first Null A
was a size-only multinomial network; it is nearly complete with rank-1
weights, Infomap never finds more than one module in it, and a null that
cannot fail cannot tell you anything. It was replaced by the degree-
preserving rewiring null, which does produce structure by chance (savings
0.06-0.10) and is what the observed values now have to beat. And our first
writeup only quoted the comparison numbers where Infomap beats Louvain;
the ARI column in `communities_metrics.csv` goes the other way in 5 of 6
periods, and the writeup now says so.
