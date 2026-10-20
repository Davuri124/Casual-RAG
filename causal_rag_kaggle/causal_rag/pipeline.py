"""Stage runners that glue the agents together (each notebook calls one of these)."""
import json
import pickle
import time
import numpy as np
import pandas as pd

from . import paths
from .config import TREATMENTS, Featurizer, fit_medians, add_features, RAW_FIELDS
from .dr_eval import crossfit_dr, overlap_report, dr_mse, blp, gates
from .prompts import card, SYSTEM_BUILDERS


# =============================================================================== Causal Attribution Agent (CALF)
def run_calf_stage(df, adj, seeds=(0, 1, 2), variant_seeds=(0,), epochs=40, with_econml=False, verbose=True, reuse=True):
    """
    Per clause: fit CALF on the tier-rule adjustment set ('primary') and compare with
      CALF K=1 (paper-as-written), flat MLP (no layers), LightGBM T-learner, EconML CausalForestDML,
      and CALF on the graph-derived adjustment set (adjustment-set sensitivity).
    Metrics are doubly-robust, cross-fitted, on the 2015 TEST cohort (no ground-truth ITE exists in real data).
    Saves calf_scores.parquet / calf_models.pt / featurizers.pkl / calf_results.csv.
    """
    import torch
    from .calf import fit_ensemble, predict_ensemble, ensemble_outputs, save_ensemble
    if reuse and paths.have("calf_scores.parquet", "calf_results.csv", "calf_models.pt", "featurizers.pkl"):
        print("CALF stage outputs already present -> reusing (delete them to retrain)")
        return pd.read_csv(paths.art("calf_results.csv")), pd.read_parquet(paths.art("calf_scores.parquet"))
    t0 = time.time()
    tr, va, te = [(df.role == r).values for r in ("train", "val", "test")]
    medians = fit_medians(df, tr)
    d = add_features(df, medians)
    y = d.default.values
    scores = {"case_id": d.case_id.values}
    res, fzs, store = [], {"medians": medians}, {}
    for tk, spec in TREATMENTS.items():
        t_tk = time.time()
        fz = Featurizer(tk).fit(d, tr)
        L, T = fz.transform(d), fz.treatment(d)
        flat = np.hstack(L)
        dr_va = crossfit_dr(flat[va], T[va], y[va])
        dr_te = crossfit_dr(flat[te], T[te], y[te])
        ov, keep = overlap_report(dr_te["e"])
        if verbose:
            print(f"\n=== {tk}: {spec['clause']} | treated share {T.mean():.3f} | layer dims {[x.shape[1] for x in L]} | overlap {ov}")

        def record(name, tau, aset, ref=None, dr=dr_te):
            c = blp(tau[keep], dr["tau_dr"][keep])
            res.append(dict(treatment=tk, model=name, adjustment_set=aset, dr_mse=dr_mse(tau[keep], dr["tau_dr"][keep]),
                            ate_dr=c["ate"], ate_dr_se=c["ate_se"], blp_slope=c["slope"], blp_slope_se=c["slope_se"],
                            corr_with_primary=float(np.corrcoef(tau, ref)[0, 1]) if ref is not None else np.nan,
                            overlap_outside=ov["share_outside"]))

        # ---- primary CALF (layered, K=4 tokens), ensemble
        sl = lambda LL, m: [x[m] for x in LL]
        prim = fit_ensemble(sl(L, tr), T[tr], y[tr], sl(L, va), dr_va["tau_dr"], seeds=seeds, epochs=epochs, verbose=False, k_tokens=4)
        out_all = ensemble_outputs(prim, L)
        tau_p = out_all["tau"][te]
        record("CALF (K=4 tokens, layered) [primary]", tau_p, "tier-rule")
        for col in ("e", "p0", "p1", "tau", "tau_sd"):
            scores[f"{col}_{tk}"] = out_all[col]
        scores[f"pf_{tk}"] = np.where(T == 1, out_all["p1"], out_all["p0"])
        save_ensemble(prim, store, tk, dict(dims=[x.shape[1] for x in L], h=128, k_tokens=4, heads=4, p=0.3, use_attn=True))
        fzs[tk] = fz

        # ---- variants (fewer seeds)
        m = fit_ensemble(sl(L, tr), T[tr], y[tr], sl(L, va), dr_va["tau_dr"], seeds=variant_seeds, epochs=epochs, verbose=False, k_tokens=1)
        record("CALF K=1 (paper-as-written attention)", predict_ensemble(m, sl(L, te))[0], "tier-rule", tau_p)
        F1 = [flat]
        m = fit_ensemble(sl(F1, tr), T[tr], y[tr], sl(F1, va), dr_va["tau_dr"], seeds=variant_seeds, epochs=epochs, verbose=False)
        record("Flat MLP (no layers, no attention)", predict_ensemble(m, sl(F1, te))[0], "tier-rule", tau_p)

        import lightgbm as lgb
        g = lambda: lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, min_child_samples=100, verbose=-1)
        m1 = g().fit(flat[tr & (T == 1)], y[tr & (T == 1)]); m0 = g().fit(flat[tr & (T == 0)], y[tr & (T == 0)])
        record("LightGBM T-learner", (m1.predict_proba(flat[te])[:, 1] - m0.predict_proba(flat[te])[:, 1]), "tier-rule", tau_p)

        if with_econml:
            try:
                from econml.dml import CausalForestDML
                rng = np.random.default_rng(0)
                sub = rng.choice(np.where(tr)[0], min(20_000, tr.sum()), replace=False)   # forests are CPU-heavy: keep it small
                cf = CausalForestDML(model_y=lgb.LGBMRegressor(n_estimators=150, verbose=-1),   # binary y is regressed (EconML requirement)
                                     model_t=lgb.LGBMClassifier(n_estimators=150, verbose=-1),
                                     discrete_treatment=True, n_estimators=100, min_samples_leaf=50, cv=2, inference=False, random_state=0)
                cf.fit(y[sub], T[sub], X=flat[sub])
                record("EconML CausalForestDML", cf.effect(flat[te]), "tier-rule", tau_p)
            except Exception as e:
                print("  (EconML skipped:", type(e).__name__, str(e)[:80], ")")

        # ---- adjustment-set sensitivity: graph-derived set (removes possible mediators / bad controls)
        fzg = Featurizer(tk, extra_drop=adj[tk]["graph_drop"]).fit(d, tr)
        Lg = fzg.transform(d)
        flat_g = np.hstack(Lg)
        dr_va_g, dr_te_g = crossfit_dr(flat_g[va], T[va], y[va]), crossfit_dr(flat_g[te], T[te], y[te])
        m = fit_ensemble(sl(Lg, tr), T[tr], y[tr], sl(Lg, va), dr_va_g["tau_dr"], seeds=variant_seeds, epochs=epochs, verbose=False, k_tokens=4)
        record("CALF K=4 on graph-derived adjustment set", predict_ensemble(m, sl(Lg, te))[0], "graph-derived", tau_p, dr=dr_te_g)
        if verbose:
            print(f"[{tk}] done in {(time.time() - t_tk) / 60:.1f} min (total {(time.time() - t0) / 60:.1f} min)")
            print(pd.DataFrame(res[-6:]).round(4).to_string(index=False))
            print("quintile calibration (primary):\n", gates(tau_p[keep], dr_te["tau_dr"][keep]).round(4))

    sc = pd.DataFrame(scores)
    sc["role"] = df.role.values
    sc.to_parquet(paths.art("calf_scores.parquet"))
    torch.save(store, paths.art("calf_models.pt"))
    pickle.dump(fzs, open(paths.art("featurizers.pkl"), "wb"))
    out = pd.DataFrame(res).round(5)
    out.to_csv(paths.art("calf_results.csv"), index=False)
    return out, sc


# =============================================================================== online CALF (demo / what-if)
class CalfScorer:
    def __init__(self):
        import torch
        from .calf import load_ensemble
        self.fz = pickle.load(open(paths.art("featurizers.pkl"), "rb"))
        store = torch.load(paths.art("calf_models.pt"), weights_only=False)
        self.models = {tk: load_ensemble(store[tk]) for tk in TREATMENTS}

    def score(self, a):
        from .calf import ensemble_outputs
        row = {f: a[f] for f in RAW_FIELDS}
        d = add_features(pd.DataFrame([row]), self.fz["medians"])
        out = {}
        for tk in TREATMENTS:
            o = ensemble_outputs(self.models[tk], self.fz[tk].transform(d))
            out[tk] = {k: float(v[0]) for k, v in o.items()}
        return out


# =============================================================================== embeddings
def build_case_index(df, embedder, verbose=True):
    """Embed cards of labelled historical loans (train+val). Text excludes employer/loan titles (noise)."""
    from .retrieval import row_to_a
    d = df[df.role.isin(["train", "val"])].reset_index(drop=True)
    texts = [card(row_to_a(r), with_text=False) for _, r in d.iterrows()]
    emb = embedder.encode(texts, bs=512, progress=verbose)
    np.save(paths.art("case_emb.npy"), emb.astype("float16"))
    d[["case_id"]].to_parquet(paths.art("case_cards.parquet"))
    if verbose:
        print("case index:", emb.shape, "| example card:\n", texts[0])
    return emb


def build_policy_index(embedder, chunks):
    emb = embedder.encode(chunks.text.tolist(), bs=64)
    np.save(paths.art("policy_emb.npy"), emb.astype("float16"))
    return emb


# =============================================================================== scoring
def score_systems(llm, bundles, systems=("no_rag", "vanilla_rag", "knn_rag", "causal_rag"), batch_size=4):
    out = {}
    for s in systems:
        prompts = [SYSTEM_BUILDERS[s](b["a"], b) for b in bundles]
        out[s] = llm.margins(prompts, batch_size=batch_size)
    return out


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =============================================================================== resumable LLM stage
def run_llm_stage(llm, bundles, idx_df, base_systems, cf_systems, boundary_systems=("causal_rag", "causal_rag_smooth"),
                  chunk=100, resume=True, verbose=True, tag="llm_parts", boundary_only=()):
    """
    Scores every system and runs the Counterfactual Perturbation Agent in CHUNKS of applicants. Each chunk is written
    to ART/llm_parts/ as soon as it finishes; a re-run (same bundles) resumes from the first missing chunk, so a crashed
    or timed-out session never costs more than one chunk.
    """
    from .counterfactual import CounterfactualAgent
    import time
    parts = paths.ART / tag
    parts.mkdir(exist_ok=True)
    agent = CounterfactualAgent(llm, idx_df, systems=tuple(cf_systems), boundary_systems=tuple(boundary_systems),
                                boundary_only=boundary_only)
    base_all, cf_all = [], []
    for k, s in enumerate(range(0, len(bundles), chunk)):
        sub = bundles[s:s + chunk]
        fb, fc = parts / f"base_{k:03d}.parquet", parts / f"cf_{k:03d}.parquet"
        ids = [b["case_id"] for b in sub]
        if resume and fb.exists() and fc.exists() and pd.read_parquet(fb).case_id.tolist() == ids:
            base_df, cf_df = pd.read_parquet(fb), pd.read_parquet(fc)
            if verbose:
                print(f"chunk {k}: loaded from disk")
        else:
            t0 = time.time()
            base = score_systems(llm, sub, base_systems)
            cf_df = agent.run(sub, base)
            base_df = pd.DataFrame({"case_id": ids, **base})
            base_df.to_parquet(fb); cf_df.to_parquet(fc)
            if verbose:
                print(f"chunk {k}: {len(sub)} applicants in {(time.time() - t0) / 60:.1f} min")
        base_all.append(base_df); cf_all.append(cf_df)
    return pd.concat(base_all, ignore_index=True), pd.concat(cf_all, ignore_index=True)


def run_cliff_stage(llm, bundles, systems, chunk=50, resume=True, tag="cliff_parts", verbose=True):
    """Chunked, resumable local-cliff test (see counterfactual.CounterfactualAgent.run_local_cliff)."""
    from .counterfactual import CounterfactualAgent
    import time
    parts = paths.ART / tag
    parts.mkdir(exist_ok=True)
    agent = CounterfactualAgent(llm, None, systems=tuple(systems), boundary_systems=())
    out = []
    for k, s in enumerate(range(0, len(bundles), chunk)):
        sub = bundles[s:s + chunk]
        f = parts / f"cliff_{k:03d}.parquet"
        ids = [b["case_id"] for b in sub]
        if resume and f.exists() and set(pd.read_parquet(f).case_id) <= set(ids):
            w = pd.read_parquet(f)
            if verbose: print(f"chunk {k}: loaded from disk")
        else:
            t0 = time.time()
            w = agent.run_local_cliff(sub, systems)
            w.to_parquet(f)
            if verbose: print(f"chunk {k}: {len(sub)} applicants in {(time.time() - t0) / 60:.1f} min")
        out.append(w)
    return pd.concat(out, ignore_index=True)
