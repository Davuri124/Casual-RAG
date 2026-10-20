"""
Stage 1 - real Lending Club data -> clean, leakage-free, time-split case table.

Design decisions (all deliberate, all reported):
  * Only PRE-ORIGINATION columns. No payment history / recoveries (post-outcome leakage).
  * grade / sub_grade / int_rate are EXCLUDED: they are the lender's own risk output (a mediator /
    the old decision). Including them would leak the previous underwriting decision into the "causes".
  * 36-month loans issued 2012-2015 only, resolved outcomes only (Fully Paid / Charged Off).
  * Right-censoring guard: for each issue month we measure the share of loans still unresolved
    (Current / Late / Grace). The test cohort is cut at the first month where it exceeds 1%, otherwise
    the surviving "resolved" loans would be biased towards early payoffs/early defaults.
  * Time split: train 2012-2014 (15% of it held out as `val`), test = 2015 (up to the censoring cutoff).
"""
import numpy as np
import pandas as pd

from . import paths

USECOLS = ["loan_status", "issue_d", "term", "loan_amnt", "emp_length", "home_ownership", "annual_inc",
           "verification_status", "purpose", "addr_state", "dti", "delinq_2yrs", "fico_range_low",
           "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
           "total_acc", "mort_acc", "earliest_cr_line", "emp_title", "title"]
RESOLVED = {"Fully Paid": 0, "Charged Off": 1}


def find_lending_club():
    p = paths.find_input("accepted*.csv*")
    if p is None:
        raise FileNotFoundError(
            "Lending Club accepted-loans file not found under /kaggle/input. Attach the Kaggle dataset "
            "'wordsforthewise/lending-club' (Add Input -> search 'Lending Club').")
    return p


def prepare(path=None, smoke=False, seed=42, verbose=True):
    path = path or find_lending_club()
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    df["issue_d"] = pd.to_datetime(df.issue_d, format="%b-%Y")
    df["term"] = df.term.astype(str).str.extract(r"(\d+)")[0].astype(float)
    df = df[(df.term == 36) & df.issue_d.dt.year.between(2012, 2015)].copy()

    # ---- censoring report (over ALL statuses, before resolving filter)
    st = df.loan_status.astype(str)
    df["_resolved"] = st.isin(RESOLVED)
    df["_inflight"] = ~df._resolved & ~st.str.startswith("Does not meet")
    rep = (df.groupby(df.issue_d.dt.to_period("M"))
             .agg(n=("loan_status", "size"), resolved=("_resolved", "mean"), inflight=("_inflight", "mean")))
    bad = rep[(rep.index.year == 2015) & (rep.inflight > 0.01)]
    cutoff = bad.index.min().to_timestamp() if len(bad) else pd.Timestamp("2016-01-01")
    rep.to_csv(paths.art("censoring_report.csv"))
    if verbose:
        print("In-flight share by issue month (2015):\n", rep[rep.index.year == 2015].round(4).to_string())
        print("Test cohort cutoff (exclusive):", cutoff.date())

    df = df[df._resolved & (df.issue_d < cutoff)].copy()
    df["default"] = df.loan_status.map(RESOLVED).astype(int)

    # ---- features known at application time
    df["fico"] = (df.fico_range_low + df.fico_range_high) / 2
    df["credit_age_yrs"] = (df.issue_d - pd.to_datetime(df.earliest_cr_line, format="%b-%Y")).dt.days / 365.25
    el = df.emp_length.astype(str)
    df["emp_years"] = np.where(el.str.contains("<"), 0.0,
                               pd.to_numeric(el.str.extract(r"(\d+)")[0], errors="coerce"))
    df["emp_title"] = df.emp_title.fillna("").astype(str).str.strip().str.lower().str[:40]
    df["title"] = df.title.fillna("").astype(str).str.strip().str.lower().str[:40]
    df = df.drop(columns=["fico_range_low", "fico_range_high", "earliest_cr_line", "emp_length",
                          "loan_status", "term", "_resolved", "_inflight"])
    df = df[(df.annual_inc > 0) & df.dti.between(0, 60)].dropna(
        subset=["dti", "revol_util", "fico", "credit_age_yrs", "home_ownership", "purpose",
                "addr_state", "verification_status"])
    df = df[df.home_ownership.isin(["RENT", "OWN", "MORTGAGE"])]       # drop ANY/NONE/OTHER (tiny, unstable)

    # ---- roles
    rng = np.random.default_rng(seed)
    is_test = df.issue_d.dt.year == 2015
    df["role"] = np.where(is_test, "test", np.where(rng.random(len(df)) < 0.15, "val", "train"))
    if smoke:
        df = pd.concat([df[df.role == r].sample(min((df.role == r).sum(), 15000), random_state=seed)
                        for r in ("train", "val", "test")])
    df = df.sort_values("issue_d").reset_index(drop=True).rename_axis("case_id").reset_index()
    df["issue_d"] = df.issue_d.dt.strftime("%Y-%m")
    if verbose:
        print(df.groupby("role").agg(n=("default", "size"), default_rate=("default", "mean")).round(4))
    df.to_parquet(paths.art("lc_clean.parquet"))
    return df
