"""
Validate the effect estimator on REAL randomised experiments, where treatment assignment is random and the
propensity is known, so the experimental difference-in-means is a trustworthy reference.

  Hillstrom e-mail campaign (MineThatData 2008): 64k customers; treatment = any e-mail vs none; outcome = site visit.
  Criteo uplift (subsample)                     : 12 anonymised features, no meaningful layers -> flat model only.

What is checked
  * ATE: model-average effect vs experimental diff-in-means (95% CI).
  * Ranking: Qini coefficient on held-out data (uses real outcomes + randomisation, no model assumptions).
  * Heterogeneity calibration: BLP slope and quintile (GATES) table against IPW/DR pseudo-outcomes with KNOWN e.
CALF's layer structure is NOT expected to help on these data (the paper's insurance result says layering only
pays off when covariates really decompose into causal layers); the comparison K=4 vs K=1 vs flat reports that honestly.
"""
import io
import numpy as np
import pandas as pd

from . import paths
from .calf import fit_ensemble, predict_ensemble
from .dr_eval import crossfit_dr, blp, gates, dr_mse

HILLSTROM_URL = ("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_"
                 "DataMiningChallenge_2008.03.20.csv")


HILLSTROM_REQ = {"recency", "history", "zip_code", "newbie", "channel", "mens", "womens", "segment", "visit"}


def _std_hillstrom(d):
    d = d.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "segment" not in d.columns and "treatment" in d.columns:
        t = d["treatment"]
        d["segment"] = t if t.dtype == object else np.where(t.astype(float) > 0, "E-Mail", "No E-Mail")
    return d


def load_hillstrom():
    """Return the REAL Hillstrom table. Every candidate (attached files, URL, scikit-uplift) is validated by columns,
    so look-alike files (e.g. your own *_hillstrom_*_results.csv) are skipped instead of silently used."""
    tried = []
    cands = (paths.find_input("*illstrom*.csv", first=False) or []) + (paths.find_input("*MineThatData*.csv", first=False) or [])
    for p in cands:
        try:
            d = _std_hillstrom(pd.read_csv(p))
        except Exception as e:
            tried.append((p, type(e).__name__)); continue
        if HILLSTROM_REQ <= set(d.columns):
            print("Hillstrom loaded from attached file:", p); return d
        tried.append((p, "missing columns " + str(sorted(HILLSTROM_REQ - set(d.columns)))))
    try:
        d = _std_hillstrom(pd.read_csv(HILLSTROM_URL))
        if HILLSTROM_REQ <= set(d.columns):
            print("Hillstrom loaded from minethatdata.com"); return d
        tried.append((HILLSTROM_URL, "missing columns"))
    except Exception as e:
        tried.append((HILLSTROM_URL, type(e).__name__))
    try:
        from sklift.datasets import fetch_hillstrom
        x = fetch_hillstrom(target_col="visit")
        d = x.data.copy(); d["segment"] = x.treatment.values; d["visit"] = x.target.values
        d = _std_hillstrom(d)
        if HILLSTROM_REQ <= set(d.columns):
            print("Hillstrom loaded via scikit-uplift"); return d
        tried.append(("scikit-uplift", "missing columns"))
    except Exception as e:
        tried.append(("scikit-uplift", type(e).__name__))
    raise RuntimeError("Could not obtain the Hillstrom data. Tried: %s. Attach a Kaggle dataset containing the original "
                       "Hillstrom e-mail CSV, or turn Internet on." % tried)


def hillstrom_layers(df):
    d = df.copy()
    T = (d.segment != "No E-Mail").astype("float32").values
    y = d.visit.astype("float32").values
    def z(x):
        x = np.asarray(x, "float32"); return ((x - x.mean()) / (x.std() + 1e-6))
    roots = np.c_[z(d.recency), z(np.log1p(d.history))]
    soil = pd.get_dummies(d.zip_code, dtype="float32").values
    leaves = np.c_[z(d.newbie), pd.get_dummies(d.channel, dtype="float32").values]
    choice = np.c_[d.mens.values, d.womens.values].astype("float32")
    return [roots.astype("float32"), soil, leaves.astype("float32"), choice], T, y


def load_criteo(n=400_000, seed=0):
    p = paths.find_input("*riteo*.csv*")
    if p:
        d = pd.read_csv(p)
    else:
        from sklift.datasets import fetch_criteo
        x = fetch_criteo(target_col="visit", treatment_col="treatment", percent10=True)
        d = x.data.copy(); d["treatment"] = x.treatment.values; d["visit"] = x.target.values
    return d.sample(min(n, len(d)), random_state=seed)


def qini_auc(tau_hat, T, y, grid=100):
    """Normalised Qini coefficient (area between model curve and random-targeting line), real outcomes only."""
    o = np.argsort(-tau_hat)
    T, y = T[o], y[o]
    n = len(y); cum = np.linspace(0, n, grid + 1).astype(int)[1:]
    nt, nc = np.cumsum(T), np.cumsum(1 - T)
    yt, yc = np.cumsum(y * T), np.cumsum(y * (1 - T))
    q = yt[cum - 1] - yc[cum - 1] * nt[cum - 1] / np.maximum(nc[cum - 1], 1)
    qr = q[-1] * cum / n
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return float(trap(q - qr, cum / n))


def evaluate_rct(layers, T, y, known_e, seed=0, seeds=(0, 1, 2), epochs=25, flat_only=False):
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr, va, te = perm[: int(.7 * n)], perm[int(.7 * n): int(.85 * n)], perm[int(.85 * n):]
    flat = np.hstack(layers)
    dr_va = crossfit_dr(flat[va], T[va], y[va], known_e=known_e)["tau_dr"]
    dr_te = crossfit_dr(flat[te], T[te], y[te], known_e=known_e)["tau_dr"]
    ate_exp = y[te][T[te] == 1].mean() - y[te][T[te] == 0].mean()
    se_exp = np.sqrt(y[te][T[te] == 1].var(ddof=1) / (T[te] == 1).sum() + y[te][T[te] == 0].var(ddof=1) / (T[te] == 0).sum())
    variants = {"CALF flat (1 layer)": ([flat], dict(k_tokens=4))}
    if not flat_only:
        variants = {"CALF K=4 tokens (layered)": (layers, dict(k_tokens=4)),
                    "CALF K=1 (paper-as-written)": (layers, dict(k_tokens=1)), **variants}
    rows = []
    for name, (L, kw) in variants.items():
        ms = fit_ensemble([x[tr] for x in L], T[tr], y[tr], [x[va] for x in L], dr_va, seeds=seeds,
                          epochs=epochs, verbose=False, **kw)
        tau, _ = predict_ensemble(ms, [x[te] for x in L])
        c = blp(tau, dr_te)
        rows.append(dict(model=name, ate_model=float(tau.mean()), ate_experiment=float(ate_exp),
                         ate_exp_ci=f"[{ate_exp - 1.96 * se_exp:.4f}, {ate_exp + 1.96 * se_exp:.4f}]",
                         ate_in_ci=bool(abs(tau.mean() - ate_exp) < 1.96 * se_exp), dr_mse=dr_mse(tau, dr_te),
                         blp_slope=c["slope"], blp_se=c["slope_se"], qini=qini_auc(tau, T[te], y[te])))
        if name.startswith("CALF flat") or name.startswith("CALF K=4"):
            last = (name, gates(tau, dr_te))
    # LightGBM T-learner reference
    import lightgbm as lgb
    g = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1)
    m1 = g().fit(flat[tr][T[tr] == 1], y[tr][T[tr] == 1]); m0 = g().fit(flat[tr][T[tr] == 0], y[tr][T[tr] == 0])
    tau = m1.predict_proba(flat[te])[:, 1] - m0.predict_proba(flat[te])[:, 1]
    c = blp(tau, dr_te)
    rows.append(dict(model="LightGBM T-learner", ate_model=float(tau.mean()), ate_experiment=float(ate_exp),
                     ate_exp_ci=f"[{ate_exp - 1.96 * se_exp:.4f}, {ate_exp + 1.96 * se_exp:.4f}]",
                     ate_in_ci=bool(abs(tau.mean() - ate_exp) < 1.96 * se_exp), dr_mse=dr_mse(tau, dr_te),
                     blp_slope=c["slope"], blp_se=c["slope_se"], qini=qini_auc(tau, T[te], y[te])))
    return pd.DataFrame(rows).round(5), last
