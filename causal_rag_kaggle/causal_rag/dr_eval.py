"""
Evaluation WITHOUT ground-truth individual effects (real data).

PEHE needs tau_i, which is never observed. Instead we use a cross-fitted
doubly-robust (AIPW) pseudo-outcome:
        tau_dr = mu1 - mu0 + (T - e) / (e (1 - e)) * (y - mu_T)
E[tau_dr | X] = tau(X) if EITHER the propensity or the outcome nuisances are right, so
        E[(tau_hat - tau_dr)^2] = E[(tau_hat - tau)^2] + const
and the constant is identical across models => DR-MSE ranks models like PEHE would.
Absolute DR-MSE values are NOT comparable to a paper PEHE.

Compute pseudo-outcomes separately inside each evaluation split (val, test).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _gbm(n_estimators=300, **kw):
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              min_child_samples=50, verbose=-1, **kw)


def crossfit_dr(X, T, y, n_splits=5, seed=0, clip=0.02, known_e=None, n_estimators=300):
    """X: flat design matrix (hstack of the layers). Returns dict(tau_dr, e, mu0, mu1)."""
    T = np.asarray(T).astype(int); y = np.asarray(y).astype(int)
    n = len(y)
    e, m0, m1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, T * 2 + y):
        if known_e is None:
            e[te] = _gbm(n_estimators, random_state=seed).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
        else:                                                   # randomised experiment: propensity is KNOWN
            e[te] = known_e
        for t, m in ((0, m0), (1, m1)):
            idx = tr[T[tr] == t]
            m[te] = _gbm(n_estimators, random_state=seed).fit(X[idx], y[idx]).predict_proba(X[te])[:, 1]
    e = np.clip(e, clip, 1 - clip)
    mu_T = np.where(T == 1, m1, m0)
    tau_dr = m1 - m0 + (T - e) / (e * (1 - e)) * (y - mu_T)
    return dict(tau_dr=tau_dr, e=e, mu0=m0, mu1=m1)


def overlap_report(e, lo=0.05, hi=0.95):
    """Positivity diagnostic. Report it; evaluate on the trimmed set if the tails are heavy."""
    keep = (e > lo) & (e < hi)
    return dict(share_outside=float(1 - keep.mean()), e_q01=float(np.quantile(e, .01)),
                e_q99=float(np.quantile(e, .99))), keep


def dr_mse(tau_hat, tau_dr):
    return float(np.mean((tau_hat - tau_dr) ** 2))


def blp(tau_hat, tau_dr):
    """
    Best-linear-predictor calibration test:  tau_dr = a + b * (tau_hat - mean) + err.
    a  = ATE estimate (with HC0 SE).
    b ~ 1  -> predicted heterogeneity is real and well scaled;  b ~ 0 -> no heterogeneity signal;
    b > 1 -> model under-disperses; 0 < b < 1 -> over-disperses.
    """
    x = tau_hat - tau_hat.mean()
    X = np.c_[np.ones_like(x), x]
    b = np.linalg.lstsq(X, tau_dr, rcond=None)[0]
    r = tau_dr - X @ b
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ ((X * r[:, None]).T @ (X * r[:, None])) @ bread
    se = np.sqrt(np.diag(cov))
    return dict(ate=b[0], ate_se=se[0], slope=b[1], slope_se=se[1])


def gates(tau_hat, tau_dr, q=5):
    """Group ATEs by predicted-effect quantile. Observed (DR) column should rise with the predicted column."""
    rows = []
    for b in np.array_split(np.argsort(tau_hat), q):
        d = tau_dr[b]
        rows.append(dict(n=len(b), predicted=tau_hat[b].mean(), observed_dr=d.mean(),
                         se=d.std(ddof=1) / np.sqrt(len(b))))
    return pd.DataFrame(rows)


def bootstrap_ci(tau_dr, idx_groups=None, B=500, seed=0):
    """Percentile CI for the mean DR effect (whole sample, or per index group)."""
    rng = np.random.default_rng(seed)
    groups = {"all": np.arange(len(tau_dr))} if idx_groups is None else idx_groups
    out = {}
    for k, ix in groups.items():
        ms = [tau_dr[rng.choice(ix, len(ix))].mean() for _ in range(B)]
        out[k] = (tau_dr[ix].mean(), *np.quantile(ms, [.025, .975]))
    return out
