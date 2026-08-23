#!/usr/bin/env python
"""
05_regression.py
================
Attach macro-level indicators to the instance-level frame scores and estimate
the Long-term and Modern OLS models for each frame.

Two model specifications per frame (dependent variable = baseline-corrected
frame score, top{N}_{c}_rel_pll):

    Long-term : score ~ v2x_libdem + pre_1960
                (full period; pre_1960 dummy absorbs pre-mid-century structure)
    Modern    : score ~ v2x_libdem + log_gdp + wdi_trade + log_migrant
                (post-1960 subset with complete economic data)

Estimated at two levels:
    - instance level, standard errors clustered by year   (main table)
    - year level (annual means), HAC standard errors      (robustness)

Because the regression requires complete macro-level indicators, instances
from years lacking the relevant values are dropped listwise; this is what
reduces the long-term sample below the full instance count.

Inputs
------
    --scores   scores.csv         (from 04; instance-level frame scores)
    --vdem     V-Dem CSV          (country-year; filtered to the U.S.)
    --macro    macro indicators CSV keyed by `year` (WDI/QoG etc.), containing
               at least: wdi_gdpcapcon2015_k, wdi_trade, wdi_migration_10k

Output
------
Prints both regression tables and (optionally) writes them to text files.

Example
-------
    python 05_regression.py \
        --scores scores.csv \
        --vdem V-Dem-CD-v15.csv \
        --macro macro_indicators.csv \
        --top-n 3 \
        --out-prefix regression
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

CATS = ["nation", "civic", "global"]


def load_vdem(path, indicators=("v2x_libdem", "v2x_polyarchy", "v2x_partipdem")):
    df = pd.read_csv(path)
    df = df[df["country_name"] == "United States of America"].reset_index(drop=True)
    keep = ["year"] + [c for c in indicators if c in df.columns]
    df = df[keep].drop_duplicates(subset="year", keep="first")
    return df


def prepare(df):
    """Add logged covariates and the pre-1960 dummy."""
    df = df.copy()
    df["log_gdp"] = np.log(df["wdi_gdpcapcon2015_k"] + 1)
    df["log_migrant"] = np.log(df["wdi_migration_10k"] + 1)
    df["pre_1960"] = (df["year"] < 1960).astype(int)
    return df


def run_ols_table(df, dep_vars, cov_type, cov_kwds_long, cov_kwds_modern,
                  group_col=None, title=""):
    """Estimate Long/Modern OLS for each dependent variable and tabulate."""
    df_modern_base = df.dropna(subset=["wdi_gdpcapcon2015_k", "wdi_migration_10k"])

    results, names = [], []
    for dep in dep_vars:
        # Long-term
        cols_l = [dep, "v2x_libdem", "pre_1960"] + ([group_col] if group_col else [])
        d_l = df.dropna(subset=cols_l).reset_index(drop=True)
        kw_l = dict(cov_type=cov_type)
        if cov_type == "cluster":
            kw_l["cov_kwds"] = {"groups": d_l[group_col]}
        else:
            kw_l["cov_kwds"] = cov_kwds_long
        res_l = smf.ols(f"{dep} ~ v2x_libdem + pre_1960", data=d_l).fit(**kw_l)
        results.append(res_l)

        # Modern
        cols_m = [dep, "v2x_libdem", "log_gdp", "wdi_trade", "log_migrant"] + \
                 ([group_col] if group_col else [])
        d_m = df_modern_base.dropna(subset=cols_m).reset_index(drop=True)
        kw_m = dict(cov_type=cov_type)
        if cov_type == "cluster":
            kw_m["cov_kwds"] = {"groups": d_m[group_col]}
        else:
            kw_m["cov_kwds"] = cov_kwds_modern
        res_m = smf.ols(
            f"{dep} ~ v2x_libdem + log_gdp + wdi_trade + log_migrant",
            data=d_m).fit(**kw_m)
        results.append(res_m)

        short = dep.split("_")[1].capitalize()   # top{N}_<cat>_rel_pll -> Cat
        names.extend([f"{short}_Long", f"{short}_Modern"])
        print(f"  [{dep}] Long N={len(d_l)}, Modern N={len(d_m)}")

    table = summary_col(
        results, model_names=names, stars=True, float_format="%0.3f",
        info_dict={"N": lambda x: f"{int(x.nobs)}",
                   "R2": lambda x: f"{x.rsquared:.3f}"})
    print(f"\n--- {title} ---")
    print(table)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="scores.csv from step 04.")
    ap.add_argument("--vdem", required=True, help="V-Dem country-year CSV.")
    ap.add_argument("--macro", required=True,
                    help="Year-keyed macro indicators CSV (WDI/QoG etc.).")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--out-prefix", default=None,
                    help="If set, write tables to <prefix>_instance.txt / _year.txt.")
    args = ap.parse_args()

    dep_vars = [f"top{args.top_n}_{c}_rel_pll" for c in CATS]

    print("[load] scores / vdem / macro")
    df_scores = pd.read_csv(args.scores)
    df_vdem = load_vdem(args.vdem)
    df_macro = pd.read_csv(args.macro)

    # ── instance level ────────────────────────────────────────────────────
    df_inst = df_scores.merge(df_vdem, on="year", how="inner")
    df_inst = df_inst.merge(df_macro, on="year", how="left")
    df_inst = prepare(df_inst)

    print("\n[instance-level, SE clustered by year]")
    tbl_inst = run_ols_table(
        df_inst, dep_vars,
        cov_type="cluster", cov_kwds_long=None, cov_kwds_modern=None,
        group_col="year",
        title="Instance-level OLS (clustered SE by year) — Long vs Modern")

    # ── year level (robustness) ───────────────────────────────────────────
    agg = {d: "mean" for d in dep_vars}
    agg.update({"v2x_libdem": "first"})
    df_year = df_inst.groupby("year", as_index=False).agg(agg)
    # re-attach macro at the year level for the modern spec
    df_year = df_year.merge(
        df_macro, on="year", how="left")
    df_year = prepare(df_year)

    print("\n[year-level, HAC SE]")
    tbl_year = run_ols_table(
        df_year, dep_vars,
        cov_type="HAC",
        cov_kwds_long={"maxlags": 20}, cov_kwds_modern={"maxlags": 5},
        group_col=None,
        title="Year-level OLS (HAC SE) — Long vs Modern (robustness)")

    if args.out_prefix:
        with open(f"{args.out_prefix}_instance.txt", "w") as f:
            f.write(tbl_inst.as_text())
        with open(f"{args.out_prefix}_year.txt", "w") as f:
            f.write(tbl_year.as_text())
        print(f"\n[done] tables written with prefix '{args.out_prefix}'")


if __name__ == "__main__":
    main()
