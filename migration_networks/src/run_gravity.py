"""Problem 3: can economics + geography explain the flows?

PPML (Poisson pseudo-maximum likelihood) gravity model - the standard for
flow data because it handles the many zero flows (35.7% of corridor-periods)
and heteroskedasticity that break log-OLS.

Full model: ln dist, ln GDP o/d, ln pop o/d, contiguity, common official
language, colonial ties (ever), common colonizer, + period fixed effects.
Baseline: ln dist + period FE only.
Evaluation: 80/20 random holdout; Spearman/Pearson of log1p(pred) vs
log1p(actual) on test, plus deviance pseudo-R2 on train.
Residuals: actual/predicted ratio among substantial corridors -> where
does migration exceed what economics predicts?
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr
from common import load_flows, PERIODS, OUT, EST, DATA

CEPII_FIX = {"SUD": "SDN"}  # flows code -> CEPII code (unified Sudan pre-2010)


def build_dataset():
    flows = load_flows()
    flows["o"] = flows.orig.replace(CEPII_FIX)
    flows["d"] = flows.dest.replace(CEPII_FIX)
    g = pd.read_csv(f"{DATA}/gravity_dedup.csv.gz")
    g = g[g.year.isin(PERIODS)]
    assert not g.duplicated(["year", "iso3_o", "iso3_d"]).any()
    m = flows.merge(g, left_on=["year0", "o", "d"],
                    right_on=["year", "iso3_o", "iso3_d"], how="left")
    assert len(m) == len(flows), "merge inflated rows"
    need = ["dist", "gdp_o", "gdp_d", "pop_o", "pop_d",
            "contig", "comlang_off", "col_dep_ever", "comcol"]
    n0 = len(m)
    m = m.dropna(subset=need)
    m = m[(m.gdp_o > 0) & (m.gdp_d > 0) & (m.pop_o > 0) & (m.pop_d > 0)]
    print(f"rows: {n0} -> {len(m)} after covariate filter "
          f"({1 - len(m)/n0:.1%} dropped, mostly nonexistent-entity years "
          f"and micro-territories)")
    for c in ["dist", "gdp_o", "gdp_d", "pop_o", "pop_d"]:
        m[f"ln_{c}"] = np.log(m[c])
    for y in PERIODS[1:]:
        m[f"per_{y}"] = (m.year0 == y).astype(float)
    return m


FULL_X = (["ln_dist", "ln_gdp_o", "ln_gdp_d", "ln_pop_o", "ln_pop_d",
           "contig", "comlang_off", "col_dep_ever", "comcol"]
          + [f"per_{y}" for y in PERIODS[1:]])
BASE_X = ["ln_dist"] + [f"per_{y}" for y in PERIODS[1:]]


def fit_ppml(df, xcols):
    X = sm.add_constant(df[xcols])
    model = sm.GLM(df[EST], X, family=sm.families.Poisson())
    return model.fit(cov_type="HC1")


def evaluate(res, df, xcols):
    pred = res.predict(sm.add_constant(df[xcols]))
    a, p = np.log1p(df[EST]), np.log1p(pred)
    return dict(spearman=spearmanr(a, p).statistic,
                pearson_log=pearsonr(a, p).statistic)


def main():
    m = build_dataset()
    rng = np.random.default_rng(42)
    test_mask = rng.random(len(m)) < 0.2
    train, test = m[~test_mask], m[test_mask]
    print(f"train {len(train)}, test {len(test)}")

    full = fit_ppml(train, FULL_X)
    base = fit_ppml(train, BASE_X)

    coefs = pd.DataFrame({"coef": full.params, "se": full.bse}).round(4)
    coefs.to_csv(f"{OUT}/results/gravity_coefficients.csv")
    print(coefs.loc[["ln_dist", "ln_gdp_o", "ln_gdp_d", "ln_pop_o", "ln_pop_d",
                     "contig", "comlang_off", "col_dep_ever", "comcol"]])

    # Dyad-clustered SEs: 6 repeated observations per country pair make
    # HC1 too optimistic; cluster by (orig, dest). Coefficients identical.
    tr = train.copy()
    tr["dyad"] = tr.orig + "_" + tr.dest
    full_cl = sm.GLM(tr[EST], sm.add_constant(tr[FULL_X]),
                     family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": tr["dyad"]})
    pd.DataFrame({"coef": full_cl.params, "se_clustered": full_cl.bse,
                  "se_HC1": full.bse}).round(4).to_csv(
        f"{OUT}/results/gravity_coefficients_clustered.csv")

    ev_full, ev_base = evaluate(full, test, FULL_X), evaluate(base, test, BASE_X)
    ps_full = 1 - full.deviance / full.null_deviance
    ps_base = 1 - base.deviance / base.null_deviance
    summary = dict(test_spearman_full=ev_full["spearman"],
                   test_spearman_base=ev_base["spearman"],
                   test_pearson_log_full=ev_full["pearson_log"],
                   test_pearson_log_base=ev_base["pearson_log"],
                   train_pseudoR2_full=ps_full, train_pseudoR2_base=ps_base,
                   n_train=len(train), n_test=len(test))
    pd.Series(summary).to_csv(f"{OUT}/results/gravity_fit.csv")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()})

    # residuals on the FULL dataset (fit on train, applied everywhere)
    m = m.copy()
    m["pred"] = full.predict(sm.add_constant(m[FULL_X]))
    m["log_resid"] = np.log1p(m[EST]) - np.log1p(m.pred)
    m["ratio"] = (m[EST] + 1) / (m.pred + 1)
    m[["year0", "orig", "dest", EST, "pred", "log_resid", "ratio"]].to_csv(
        f"{OUT}/results/gravity_residuals.csv.gz", index=False, compression="gzip")

    big = m[m[EST] >= 50_000]
    top_under = big.nlargest(20, "log_resid")[["year0", "orig", "dest", EST, "pred", "ratio"]]
    top_over = big.nsmallest(20, "log_resid")[["year0", "orig", "dest", EST, "pred", "ratio"]]
    top_under.to_csv(f"{OUT}/results/gravity_top_underpredicted.csv", index=False)
    top_over.to_csv(f"{OUT}/results/gravity_top_overpredicted.csv", index=False)
    print("\nTOP UNDERPREDICTED (flow >> economics):")
    print(top_under.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
