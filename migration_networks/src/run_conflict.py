"""Conflict layer: do gravity residuals concentrate where wars happen?

War flag: country c has war in period y iff UCDP records any conflict-year
at intensity_level 2 (>=1000 battle deaths/yr) located in c during
[y, y+4]. Multi-location conflict-years are split to each location.

Test: origin-period mean gravity log-residual (actual vs economically
predicted outflow), war vs no-war origins, Mann-Whitney U. Positive
residuals for war origins = people leave war zones far in excess of what
economics predicts.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from common import PERIODS, OUT, DATA, EST

NAME2ISO = {
    "Afghanistan": "AFG", "Algeria": "DZA", "Angola": "AGO", "Azerbaijan": "AZE",
    "Bangladesh": "BGD", "Benin": "BEN", "Bosnia-Herzegovina": "BIH",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cambodia (Kampuchea)": "KHM",
    "Cameroon": "CMR", "Central African Republic": "CAF", "Chad": "TCD",
    "China": "CHN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
    "Croatia": "HRV", "DR Congo (Zaire)": "COD", "Djibouti": "DJI",
    "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV", "Eritrea": "ERI",
    "Ethiopia": "ETH", "Georgia": "GEO", "Ghana": "GHA", "Guatemala": "GTM",
    "Guinea": "GIN", "Guinea-Bissau": "GNB", "Haiti": "HTI", "India": "IND",
    "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Israel": "ISR",
    "Ivory Coast": "CIV", "Jordan": "JOR", "Kenya": "KEN", "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ", "Laos": "LAO", "Lebanon": "LBN", "Lesotho": "LSO",
    "Liberia": "LBR", "Libya": "LBY", "Macedonia, FYR": "MKD",
    "Madagascar (Malagasy)": "MDG", "Malaysia": "MYS", "Mali": "MLI",
    "Mauritania": "MRT", "Mexico": "MEX", "Moldova": "MDA", "Morocco": "MAR",
    "Mozambique": "MOZ", "Myanmar (Burma)": "MMR", "Nepal": "NPL",
    "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA", "Pakistan": "PAK",
    "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY",
    "Peru": "PER", "Philippines": "PHL", "Russia (Soviet Union)": "RUS",
    "Rwanda": "RWA", "Saudi Arabia": "SAU", "Senegal": "SEN",
    "Serbia (Yugoslavia)": "SRB", "Sierra Leone": "SLE", "Somalia": "SOM",
    "South Africa": "ZAF", "South Sudan": "SSD", "Spain": "ESP",
    "Sri Lanka": "LKA", "Sudan": "SDN", "Syria": "SYR", "Tajikistan": "TJK",
    "Thailand": "THA", "Togo": "TGO", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Turkey": "TUR", "Uganda": "UGA", "Ukraine": "UKR",
    "United Kingdom": "GBR", "United States of America": "USA",
    "Uzbekistan": "UZB", "Venezuela": "VEN", "Yemen (North Yemen)": "YEM",
    "Zimbabwe (Rhodesia)": "ZWE", "Tanzania": "TZA", "Mauritius": "MUS",
    "Bolivia": "BOL", "Brazil": "BRA", "Chile": "CHL", "Cuba": "CUB",
    "Dominican Republic": "DOM", "Gabon": "GAB", "Gambia": "GMB",
    "Honduras": "HND", "Malawi": "MWI", "Oman": "OMN", "Qatar": "QAT",
    "Romania": "ROU", "Suriname": "SUR", "Sweden": "SWE", "Togo ": "TGO",
    "United Arab Emirates": "ARE", "Vietnam (North Vietnam)": "VNM",
    "Zambia": "ZMB", "Estonia": "EST", "Latvia": "LVA", "Lithuania": "LTU",
    "Armenia": "ARM", "Belarus": "BLR", "Kazakhstan": "KAZ",
    "Turkmenistan": "TKM", "Albania": "ALB", "Slovenia": "SVN",
    "North Macedonia": "MKD", "Kosovo": "XKX", "Montenegro": "MNE",
    "Timor-Leste (East Timor)": "TLS", "Bhutan": "BTN",
    "Democratic Republic of Congo (Zaire)": "COD",
    "Yemen (South Yemen)": "YEM", "Yemen": "YEM", "Maldives": "MDV",
    "Solomon Islands": "SLB", "Fiji": "FJI", "Uruguay": "URY",
    "Argentina": "ARG", "Costa Rica": "CRI", "Cyprus": "CYP",
    "France": "FRA", "Greece": "GRC", "Italy": "ITA", "Japan": "JPN",
    "Netherlands": "NLD", "New Zealand": "NZL", "Norway": "NOR",
    "Portugal": "PRT", "Singapore": "SGP", "South Korea": "KOR",
    "Korea, South": "KOR", "North Korea": "PRK", "Korea, North": "PRK",
    "Australia": "AUS", "Belgium": "BEL", "Canada": "CAN", "Denmark": "DNK",
    "Germany": "DEU", "Austria": "AUT", "Switzerland": "CHE",
    "Finland": "FIN", "Ireland": "IRL", "Poland": "POL", "Hungary": "HUN",
    "Czech Republic": "CZE", "Slovakia": "SVK", "Bulgaria": "BGR",
    "Bahrain": "BHR", "Botswana": "BWA", "Equatorial Guinea": "GNQ",
    "Swaziland": "SWZ", "Eswatini": "SWZ", "Namibia": "NAM",
    "Zimbabwe": "ZWE", "Seychelles": "SYC", "Sao Tome and Principe": "STP",
}


def war_flags():
    u = pd.read_csv(f"{DATA}/UcdpPrioConflict_v26_1.csv")
    u = u[u.year >= 1990]
    rows = []
    unmapped = set()
    for loc, year, lvl in u[["location", "year", "intensity_level"]].itertuples(index=False):
        for name in [s.strip() for s in loc.split(",")]:
            iso = NAME2ISO.get(name)
            if iso is None:
                unmapped.add(name)
                continue
            rows.append((iso, year, lvl))
    assert not unmapped, f"unmapped UCDP locations: {sorted(unmapped)}"
    cy = pd.DataFrame(rows, columns=["iso3", "year", "intensity"])
    flags = []
    # In flow periods <=2005 the nodes are the predecessor states SUD and
    # SCG; UCDP-mapped ISO3 uses successor codes. Map back so flags merge.
    PRE2010 = {"SDN": "SUD", "SSD": "SUD", "SRB": "SCG", "MNE": "SCG"}
    for y in PERIODS:
        win = cy[(cy.year >= y) & (cy.year <= y + 4)]
        iso = win.iso3 if y >= 2010 else win.iso3.replace(PRE2010)
        war = set(iso[win.intensity.values == 2])
        anyc = set(iso)
        flags.append(pd.DataFrame({"period": y,
                                   "iso3": sorted(anyc | war),
                                   "war": [c in war for c in sorted(anyc | war)],
                                   }))
    return pd.concat(flags, ignore_index=True)


def main():
    fl = war_flags()
    fl.to_csv(f"{OUT}/results/war_flags.csv", index=False)
    n_war = fl[fl.war].groupby("period").size()
    print("countries with war per period:\n", n_war.to_string())

    res = pd.read_csv(f"{OUT}/results/gravity_residuals.csv.gz")
    agg = (res.groupby(["year0", "orig"])
           .agg(actual=("da_pb_closed", "sum"), pred=("pred", "sum"),
                max_corridor_resid=("log_resid", "max"),
                mean_corridor_resid=("log_resid", "mean"))
           .reset_index().rename(columns={"year0": "period", "orig": "iso3"}))
    agg["outflow_log_ratio"] = np.log((agg.actual + 1) / (agg.pred + 1))
    m = agg.merge(fl[fl.war][["period", "iso3"]].assign(war=True),
                  on=["period", "iso3"], how="left")
    m["war"] = m.war.fillna(False).astype(bool)

    rows = {}
    for metric in ["outflow_log_ratio", "max_corridor_resid", "mean_corridor_resid"]:
        w, p = m[m.war][metric], m[~m.war][metric]
        stat = mannwhitneyu(w, p, alternative="greater")
        rows[metric] = dict(war_median=w.median(), peace_median=p.median(),
                            war_mean=w.mean(), peace_mean=p.mean(),
                            U=stat.statistic, p_value=stat.pvalue,
                            n_war=len(w), n_peace=len(p))
        print(f"{metric}: war med {w.median():.3f} vs peace {p.median():.3f}, "
              f"p={stat.pvalue:.2e}")
    pd.DataFrame(rows).T.to_csv(f"{OUT}/results/conflict_test.csv")
    m.to_csv(f"{OUT}/results/origin_period_residuals_war.csv", index=False)
    # NOTE for writeup (impediments): mean_corridor_resid runs the WRONG way -
    # war concentrates outflow into few neighbor corridors while suppressing
    # diffuse migration elsewhere, so the per-corridor mean is dominated by
    # ~200 near-zero corridors. Aggregate-then-compare metrics fix this.


if __name__ == "__main__":
    main()
