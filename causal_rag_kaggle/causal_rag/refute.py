"""
Refutation & sensitivity for the OBSERVATIONAL clause effects (real Lending Club data).

Implemented directly on the cross-fitted doubly-robust (AIPW) estimator so the same estimator that
supports CALF's evaluation is the one being stress-tested (equivalent in spirit to DoWhy's refuters):

  placebo_treatment   : permute T. A valid pipeline must return ~0 effect (CI covers 0).
  random_common_cause : append an independent noise covariate. Estimate must not move.
  subset              : re-estimate on random 80% subsets. Estimate must be stable.
  E-value             : VanderWeele-Ding. How strong would an UNMEASURED confounder have to be (risk-ratio scale,
                        with both T and Y) to explain away the estimate / its CI bound?
Passing refutations does NOT prove ignorability; it rules out gross pipeline failure. E-value quantifies
the remaining untestable assumption.
"""
import numpy as np
import pandas as pd

from .dr_eval import crossfit_dr


def _ate(X, T, y, seed, n_est=150):
    r = crossfit_dr(X, T, y, n_splits=4, seed=seed, n_estimators=n_est)
    t = r["tau_dr"]
    mu1 = (r["mu1"] + T / r["e"] * (y - r["mu1"])).mean()
    mu0 = (r["mu0"] + (1 - T) / (1 - r["e"]) * (y - r["mu0"])).mean()
    return float(t.mean()), float(t.std(ddof=1) / np.sqrt(len(t))), float(mu1), float(mu0)


def evalue(rr):
    rr = max(rr, 1e-9)
    rr = 1 / rr if rr < 1 else rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def run_refutations(X, T, y, n_rep=3, max_n=60000, seed=0, name=""):
    rng = np.random.default_rng(seed)
    if len(y) > max_n:
        i = rng.choice(len(y), max_n, replace=False); X, T, y = X[i], T[i], y[i]
    ate, se, mu1, mu0 = _ate(X, T, y, seed)
    rows = [dict(treatment=name, test="estimate", value=ate, se=se, passed=None)]
    for r in range(n_rep):
        a, s, _, _ = _ate(X, rng.permutation(T), y, seed + r + 1)
        rows.append(dict(treatment=name, test="placebo_treatment", value=a, se=s, passed=abs(a) < 1.96 * s))
    for r in range(n_rep):
        a, s, _, _ = _ate(np.c_[X, rng.normal(size=len(y))], T, y, seed + r + 11)
        rows.append(dict(treatment=name, test="random_common_cause", value=a, se=s, passed=abs(a - ate) < 1.96 * se))
    for r in range(n_rep):
        j = rng.random(len(y)) < 0.8
        a, s, _, _ = _ate(X[j], T[j], y[j], seed + r + 21)
        rows.append(dict(treatment=name, test="subset_80pct", value=a, se=s, passed=abs(a - ate) < 2 * max(s, se)))
    rr = mu1 / max(mu0, 1e-6)
    lo_rr = (mu1 - 1.96 * se) / max(mu0, 1e-6) if rr >= 1 else (mu1 + 1.96 * se) / max(mu0, 1e-6)
    ci_excl_null = (rr >= 1 and lo_rr > 1) or (rr < 1 and lo_rr < 1)
    rows.append(dict(treatment=name, test="e_value_point", value=evalue(rr), se=np.nan, passed=None))
    rows.append(dict(treatment=name, test="e_value_ci_bound", value=evalue(lo_rr) if ci_excl_null else 1.0,
                     se=np.nan, passed=None))
    return pd.DataFrame(rows)


def pass_rates(df):
    d = df[df.passed.notna()]
    return d.groupby(["treatment", "test"]).passed.mean().unstack().round(2)
