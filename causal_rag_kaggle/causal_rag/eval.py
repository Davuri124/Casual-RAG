"""Evaluation for the Causal-RAG study (all metrics computed on REAL held-out 2015 loans)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from .audit import MATERIAL, RESPONSE


def auroc(y, s):
    return float(roc_auc_score(y, s))


def paired_boot_auc(y, s1, s2, B=1000, seed=0):
    """AUROC(s1) - AUROC(s2) with a paired bootstrap over applicants. Returns (diff, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(roc_auc_score(y, s1) - roc_auc_score(y, s2)), *np.quantile(d, [.025, .975]).tolist(), float(p)


def platt_cv(y, margin, folds=5, seed=0):
    """Cross-fitted Platt scaling of an LLM margin -> calibrated default probability."""
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(margin, y):
        lr = LogisticRegression(C=10.0).fit(margin[tr, None], y[tr])
        p[te] = lr.predict_proba(margin[te, None])[:, 1]
    return p


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    e = 0.0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def discrimination_table(y, scores: dict, ref="no_rag", B=1000):
    rows = []
    for name, s in scores.items():
        s = np.asarray(s, float)
        p = platt_cv(y, s)
        d = paired_boot_auc(y, s, scores[ref], B) if name != ref else (0.0, 0.0, 0.0, 1.0)
        rows.append(dict(system=name, auroc=auroc(y, s), d_vs_ref=d[0], d_lo=d[1], d_hi=d[2], p_vs_ref=d[3],
                         brier=float(brier_score_loss(y, p)), ece=ece(y, p)))
    return pd.DataFrame(rows).round(4)


def counterfactual_table(cf, thr_by_system):
    out = []
    for sysname, g in cf.groupby("system"):
        r = dict(system=sysname)
        rf = g[(g.kind == "regime_flip") & (g.tau.abs() >= MATERIAL)]
        if len(rf):
            exp = np.sign(rf.dir * rf.tau)
            got = np.sign(rf.dP)
            resp = rf.dP.abs() >= RESPONSE
            # 'always assume a clause raises risk' predictor: how easy is the sign test?
            r["trivial_prior_agree"] = float((np.sign(rf.dir) == np.sign(rf.tau)).mean())
            r.update(n_regime_pairs=len(rf), response_rate=float(resp.mean()),
                     sign_agree_all=float((exp == got).mean()),
                     sign_agree_responders=float((exp[resp] == got[resp]).mean()) if resp.any() else np.nan)
        allrf = g[g.kind == "regime_flip"]
        if len(allrf) > 5:
            r["spearman_dP_vs_tau"] = float(spearmanr(allrf.dir * allrf.dP, allrf.tau, nan_policy="omit")[0])
            r["mean_abs_dP_regime"] = float(allrf.dP.abs().mean())
            w = [spearmanr(x.dir * x.dP, x.tau, nan_policy="omit")[0] for _, x in allrf.groupby("clause") if len(x) > 10]
            r["spearman_within_clause"] = float(np.nanmean(w)) if w else np.nan
        bf = g[g.kind == "boundary_flip"]
        if len(bf):
            r["mean_abs_dP_boundary"] = float(bf.dP.abs().mean())
            # ~1.0 => the decision jumps at the threshold as much as for a full regime change (cliff behaviour)
            r["cliff_ratio_boundary_over_regime"] = r["mean_abs_dP_boundary"] / max(r.get("mean_abs_dP_regime", np.nan), 1e-9)
        sp = g[g.kind == "spurious"]
        if len(sp):
            thr = thr_by_system[sysname]
            r["spurious_mean_abs_dP"] = float(sp.dP.abs().mean())
            r["spurious_decision_flip_rate"] = float(((sp.base_margin >= thr) != (sp.new_margin >= thr)).mean())
        rv = g[(g.kind == "reverse") & (g.local_diff.abs() >= 0.03)]
        if len(rv):
            moved = rv.dP.abs() >= RESPONSE
            r["evidence_response_rate"] = float(moved.mean())
            r["evidence_reliance_correct_sign"] = float((np.sign(rv.dP[moved]) == -np.sign(rv.local_diff[moved])).mean()) if moved.any() else np.nan
        out.append(r)
    return pd.DataFrame(out).round(4)


def coverage_risk(y, decisions):
    """Default rate among APPROVED cases overall vs after the audit removes REFER cases."""
    d = pd.DataFrame(dict(y=y, dec=decisions))
    ap_all = d[d.dec.isin(["APPROVE", "REFER_TO_HUMAN"]) & (d.dec != "DECLINE")]
    ap = d[d.dec == "APPROVE"]
    return dict(n=len(d), approve_rate=float((d.dec == "APPROVE").mean()), decline_rate=float((d.dec == "DECLINE").mean()),
                refer_rate=float((d.dec == "REFER_TO_HUMAN").mean()),
                default_rate_overall=float(d.y.mean()),
                default_rate_if_refers_approved=float(ap_all.y.mean()) if len(ap_all) else np.nan,
                default_rate_among_approved=float(ap.y.mean()) if len(ap) else np.nan,
                default_rate_among_referred=float(d[d.dec == "REFER_TO_HUMAN"].y.mean()) if (d.dec == "REFER_TO_HUMAN").any() else np.nan)


def gbm_baseline(feat_tr, y_tr, feat_va, y_va, feat_te):
    """Strong supervised tabular reference (NOT a causal model, NOT an LLM): LightGBM on the flat design matrix."""
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, min_child_samples=100, verbose=-1)
    m.fit(feat_tr, y_tr, eval_set=[(feat_va, y_va)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(feat_te)[:, 1]


def pairwise_table(y, scores, pairs, B=1000):
    """Paired-bootstrap AUROC differences for chosen (a, b) comparisons (the control experiments)."""
    rows = []
    for a, b in pairs:
        if a in scores and b in scores:
            d = paired_boot_auc(y, np.asarray(scores[a], float), np.asarray(scores[b], float), B)
            rows.append(dict(a=a, b=b, auroc_a=auroc(y, scores[a]), auroc_b=auroc(y, scores[b]),
                             diff=d[0], lo=d[1], hi=d[2], p=d[3]))
    return pd.DataFrame(rows).round(4)


def cliff_table(w, by_clause=False, B=500, seed=0):
    """cliff_index_abs = mean|cross-cutoff change| / mean|equal-width change away from the cutoff| (1 = smooth, >>1 = cliff)."""
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["system", "clause"] if by_clause else ["system"]
    for k, g in w.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        ids = g.case_id.unique()
        by = {i: x for i, x in g.groupby("case_id")}
        idx = np.arange(len(ids))
        def idx_stat(sel):
            xs = pd.concat([by[ids[j]] for j in sel])
            return xs.cross.abs().mean() / max(xs.placebo.abs().mean(), 1e-9)
        boots = [idx_stat(rng.integers(0, len(ids), len(ids))) for _ in range(B)] if len(ids) > 5 else [np.nan]
        rows.append(dict(zip(keys, k), n=len(g), mean_cross=g.cross.mean(), mean_placebo=g.placebo.mean(),
                         abs_cross=g.cross.abs().mean(), abs_placebo=g.placebo.abs().mean(),
                         cliff_index_abs=g.cross.abs().mean() / max(g.placebo.abs().mean(), 1e-9),
                         ci_lo=float(np.nanquantile(boots, .025)), ci_hi=float(np.nanquantile(boots, .975))))
    return pd.DataFrame(rows).round(4)
