"""
Project configuration for Causal-RAG on Lending Club.

CALF's four layers (Roots -> Soil -> Leaves -> Choice) are mapped to the temporal / causal tiers of a
loan application. A treatment lives in one tier; its adjustment set is every covariate in STRICTLY
EARLIER tiers plus same-tier covariates that are not part of (or mechanically tied to) the treatment
definition. Tier 4 is the outcome (default).

Thresholds are ILLUSTRATIVE clause triggers. NB03 links each one to real policy text and reports
whether any policy chunk actually supports it.
"""
import numpy as np
import pandas as pd

TIERS = {
    0: ("roots",  ["annual_inc_log", "emp_years", "emp_years_na", "credit_age_yrs", "home_ownership"]),
    1: ("soil",   ["addr_state", "verification_status"]),
    2: ("leaves", ["fico", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc",
                   "mort_acc", "mort_acc_na", "revol_bal_log", "revol_util", "dti"]),
    3: ("choice", ["loan_amnt_log", "lti", "purpose"]),
}
OUTCOME = "default"
CATEGORICAL = ["home_ownership", "addr_state", "verification_status", "purpose"]
TIER_OF = {c: k for k, (_, cols) in TIERS.items() for c in cols}
TIER_OF[OUTCOME] = 4

# Raw (human-readable) applicant fields used for cards / counterfactual edits
RAW_FIELDS = ["annual_inc", "emp_years", "credit_age_yrs", "home_ownership", "addr_state",
              "verification_status", "fico", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
              "total_acc", "mort_acc", "revol_bal", "revol_util", "dti", "loan_amnt", "purpose",
              "emp_title", "title"]
# Fields that no policy clause references -> decisions must be insensitive to them
SPURIOUS_FIELDS = ["addr_state", "emp_title", "title"]

# drop = columns removed from X because they define T or are mechanically tied to it (kills overlap otherwise)
TREATMENTS = {
    # NOTE: Fannie Mae's 36% DTI guideline binds on only ~1.5% of Lending Club loans (positivity failure observed in the
    # first real run: 85% of test loans had propensity < 0.05). 25% is an ANALYST-DEFINED cutoff with real support.
    "dti_gt25":   dict(tier=2, var="dti", thr=25.0, side="gt", drop=["dti"],
                       define=lambda d: d.dti > 25.0,
                       clause="Debt-to-income ratio above 25%",
                       query="maximum total debt-to-income DTI ratio of stable monthly income",
                       entities=["dti", "income"], cutoff_pattern=r"\b25(\.\d+)?\s?(%|percent)"),
    "util_gt75":  dict(tier=2, var="revol_util", thr=75.0, side="gt", drop=["revol_util"],
                       define=lambda d: d.revol_util > 75.0,
                       clause="Revolving credit utilization above 75%",
                       query="revolving credit line utilization credit card balances relative to limits",
                       entities=["utilization"], cutoff_pattern=r"\b75(\.\d+)?\s?(%|percent)"),
    "fico_lt680": dict(tier=2, var="fico", thr=680.0, side="lt", drop=["fico"],
                       define=lambda d: d.fico < 680.0,
                       clause="Credit score below 680",
                       query="minimum representative credit score requirement for eligibility",
                       entities=["credit_score"], cutoff_pattern=r"\b680\b"),
    "lti_gt20":   dict(tier=3, var="lti", thr=0.20, side="gt", drop=["lti", "loan_amnt_log"],
                       define=lambda d: d.lti > 0.20,
                       clause="Requested amount above 20% of annual income",
                       query="loan amount relative to income repayment ability payment burden",
                       entities=["income", "repayment"], cutoff_pattern=r"\b20(\.\d+)?\s?(%|percent)"),
}


def fit_medians(df, mask):
    return {c: float(df.loc[mask, c].median()) for c in ["emp_years", "mort_acc"]}


def add_features(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    d = df.copy()
    d["annual_inc_log"] = np.log1p(d.annual_inc)
    d["revol_bal_log"] = np.log1p(d.revol_bal)
    d["loan_amnt_log"] = np.log1p(d.loan_amnt)
    d["lti"] = d.loan_amnt / d.annual_inc
    for c in ["emp_years", "mort_acc"]:
        d[c + "_na"] = d[c].isna().astype("float32")
        d[c] = d[c].fillna(medians[c])
    return d


class Featurizer:
    """Builds the layered design matrices for one treatment. Fit on TRAIN rows only; reusable at demo time."""

    def __init__(self, tkey, extra_drop=()):
        self.tkey = tkey
        self.spec = TREATMENTS[tkey] if tkey else dict(tier=3, drop=[])   # tkey=None -> all covariates (kNN baselines)
        self.drop = set(self.spec["drop"]) | set(extra_drop)
        self.tiers = []          # list of dict(num, mu, sd, cat, dummies)

    def __getstate__(self):                 # TREATMENTS holds lambdas -> not picklable; rebuild on load
        st = dict(self.__dict__); st.pop("spec", None); return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.spec = TREATMENTS[self.tkey] if self.tkey else dict(tier=3, drop=[])

    def fit(self, d, train_mask):
        self.tiers = []
        for k in range(self.spec["tier"] + 1):
            cols = [c for c in TIERS[k][1] if c not in self.drop]
            num = [c for c in cols if c not in CATEGORICAL]
            cat = [c for c in cols if c in CATEGORICAL]
            if not num and not cat:
                continue
            x = d.loc[train_mask, num].astype("float32") if num else None
            self.tiers.append(dict(
                num=num, cat=cat,
                mu=x.mean().values if num else None,
                sd=x.std().replace(0, 1).values if num else None,
                dummies=list(pd.get_dummies(d[cat], dtype="float32").columns) if cat else []))
        return self

    def transform(self, d):
        out = []
        for t in self.tiers:
            parts = []
            if t["num"]:
                x = (d[t["num"]].astype("float32").values - t["mu"]) / t["sd"]
                parts.append(np.clip(x, -5, 5))
            if t["cat"]:
                g = pd.get_dummies(d[t["cat"]], dtype="float32").reindex(columns=t["dummies"], fill_value=0)
                parts.append(g.values)
            out.append(np.hstack(parts).astype("float32"))
        return out

    def names(self):
        return [t["num"] + t["dummies"] for t in self.tiers]

    def treatment(self, d):
        return self.spec["define"](d).astype("float32").values
