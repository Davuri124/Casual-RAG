"""
Retriever Agent - dense retrieval paired with a structural causal graph.

For each applicant and each clause (treatment T_k) it assembles an EVIDENCE bundle:

  1. Dense stage    : top-`pool` historical loans by text-embedding similarity (labels known, pre-2015 only).
  2. Causal stage   : inside that pool, pick the m nearest TREATED and m nearest CONTROL loans in the 2-D
                      balance space (logit propensity e_k, logit prognostic risk p0_k) produced by CALF's
                      heads. e_k is computed on the graph-defined adjustment set, so matching on it balances
                      exactly the confounders the causal graph says matter (Hansen 2008 prognostic/propensity
                      matching). Result: a like-for-like treated-vs-control contrast, not a bag of similar cases.
  3. Local effect   : difference of default rates between the matched arms (+ CI, + balance SMD, thin-overlap flag).
  4. Policy stage   : clause text is retrieved by dense similarity restricted to chunks whose ENTITY tags link to
                      the treatment variable (the graph's entity layer); if none, the clause is flagged
                      "analyst-defined" - never papered over.

Baselines use the same index: vanilla RAG (dense top-k cases + policy), kNN-RAG (feature-space neighbours).
"""
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import TREATMENTS, RAW_FIELDS, Featurizer
from .embedding import DenseIndex
from .prompts import card
from .policy_corpus import tag_entities

VAR_ENTITY = {"annual_inc_log": "income", "fico": "credit_score", "dti": "dti",
              "revol_util": "utilization", "lti": "repayment", "loan_amnt_log": "repayment"}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def row_to_a(row):
    a = {f: row[f] for f in RAW_FIELDS}
    for f in ("emp_years", "mort_acc"):                      # restore NaN hidden by median-fill
        if f in row and row.get(f + "_na", 0) == 1:
            a[f] = float("nan")
    for f in RAW_FIELDS:
        if isinstance(a[f], (np.floating, np.integer)):
            a[f] = float(a[f])
    return a


class RetrieverAgent:
    def __init__(self, idx_df, case_emb, policy_df, policy_emb, calf_scores, embedder, adj_sets,
                 medians=None, min_policy_sim=0.45, text_weight=0.1):
        self.df = idx_df.reset_index(drop=True)
        self.y = self.df.default.values
        self.idx = DenseIndex(case_emb)
        self.pol = policy_df.reset_index(drop=True)
        self.pol["entities"] = self.pol.text.map(tag_entities)      # re-tag with the current patterns (no need to redo stage 03)
        self.pol_idx = DenseIndex(policy_emb)
        self.emb, self.adj, self.min_sim, self.tw = embedder, adj_sets, min_policy_sim, text_weight
        self.calf = calf_scores
        c = calf_scores.loc[self.df.case_id.values]
        self.T = {k: spec["define"](self.df).values.astype(bool) for k, spec in TREATMENTS.items()}
        self.Z = {k: np.c_[logit(c[f"e_{k}"].values), logit(c[f"p0_{k}"].values)] for k in TREATMENTS}
        self.zsd = {k: self.Z[k].std(0) + 1e-6 for k in TREATMENTS}
        # balance-space neighbour structures: guarantee close like-for-like matches even when the text-similarity pool is poor
        self.nn = {}
        for k in TREATMENTS:
            Zs, T = self.Z[k] / self.zsd[k], self.T[k]
            it, ic = np.where(T)[0], np.where(~T)[0]
            self.nn[k] = (it, NearestNeighbors().fit(Zs[it]), ic, NearestNeighbors().fit(Zs[ic]))
        self.flat = Featurizer(None).fit(self.df, np.ones(len(self.df), bool))
        self.Xf = np.hstack(self.flat.transform(self.df)).astype("float32")
        self.Xf_sq = (self.Xf ** 2).sum(1)
        self._pol_cache = {}

    # ------------------------------------------------------------------ policy
    def policy_for(self, tk, k=3):
        if tk in self._pol_cache:
            return self._pol_cache[tk]
        spec = TREATMENTS[tk]
        qv = self.emb.encode([spec["query"]], query=True)
        sims = self.pol_idx.scores(qv).ravel()
        ent = self.pol.entities.apply(lambda e: bool(set(e) & set(spec["entities"]))).values
        cand = np.where(ent)[0]
        chunks, topical, top, cut = [], False, 0.0, False
        if len(cand):
            sel = cand[np.argsort(-sims[cand])[:k]]
            top = float(sims[sel[0]])
            topical = top >= self.min_sim
            chunks = [dict(chunk_id=self.pol.chunk_id[i], publisher=self.pol.publisher[i], title=self.pol.title[i],
                           text=self.pol.text[i], score=float(sims[i])) for i in sel]
        # `supported` = the retrieved policy text actually STATES this clause's cutoff (regex on the cutoff number);
        # `topical`   = policy text covers the topic, but with a different / no cutoff -> clause is analyst-defined.
        cut = any(re.search(spec["cutoff_pattern"], c["text"], flags=re.I) for c in chunks)
        parents = [VAR_ENTITY[p] for p in self.adj[tk]["parents"] if p in VAR_ENTITY]
        ctx = [self.pol.chunk_id[i] for i in np.where(self.pol.entities.apply(lambda e: bool(set(e) & set(parents))))[0][:2]]
        out = dict(supported=bool(cut), topical=bool(topical), support_score=top, chunks=chunks, graph_context=ctx)
        self._pol_cache[tk] = out
        return out

    # ------------------------------------------------------------------ causal evidence
    def _evidence(self, cand, sims, qz, tk, m, qv):
        it_all, nnt, ic_all, nnc = self.nn[tk]
        z = (qz[tk] / self.zsd[tk])[None]
        kk = 4 * m
        extra = np.r_[it_all[nnt.kneighbors(z, min(kk, len(it_all)), return_distance=False)[0]],
                      ic_all[nnc.kneighbors(z, min(kk, len(ic_all)), return_distance=False)[0]]]
        cand = np.unique(np.r_[cand, extra])
        sims = self.idx.emb[cand] @ qv
        Tc = self.T[tk][cand]
        Zc = self.Z[tk][cand] / self.zsd[tk]
        dist = np.linalg.norm(Zc - qz[tk] / self.zsd[tk], axis=1) + self.tw * (1 - sims)
        it = cand[Tc][np.argsort(dist[Tc])[:m]]
        ic = cand[~Tc][np.argsort(dist[~Tc])[:m]]
        nt, nc = len(it), len(ic)
        yt, yc = int(self.y[it].sum()), int(self.y[ic].sum())
        if nt == 0 or nc == 0:
            return dict(n_t=nt, n_c=nc, y_t=yt, y_c=yc, diff=0.0, lo=-1.0, hi=1.0, thin=True, smd=np.nan,
                        ids_t=it.tolist(), ids_c=ic.tolist())
        pt, pc = (yt + .5) / (nt + 1), (yc + .5) / (nc + 1)
        diff = yt / nt - yc / nc
        se = float(np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc))
        zt, zc = self.Z[tk][it] / self.zsd[tk], self.Z[tk][ic] / self.zsd[tk]
        smd = float(np.max(np.abs(zt.mean(0) - zc.mean(0)) / np.sqrt((zt.var(0) + zc.var(0)) / 2 + 1e-9)))
        return dict(n_t=nt, n_c=nc, y_t=yt, y_c=yc, diff=float(diff), lo=float(diff - 1.96 * se),
                    hi=float(diff + 1.96 * se), thin=bool(min(nt, nc) < m // 2 or smd > 0.5), smd=smd,
                    ids_t=it.tolist(), ids_c=ic.tolist())

    def _card_of(self, i, outcome=True):
        return card(row_to_a(self.df.iloc[i]), outcome=int(self.y[i]) if outcome else None, with_text=False)

    # ------------------------------------------------------------------ bundles
    def build_bundles(self, qdf, pool=2000, m=25, k_cases=6, k_policy=3, progress=True):
        """qdf: query rows (derived features present) whose case_ids exist in calf_scores."""
        qa = [row_to_a(r) for _, r in qdf.iterrows()]
        qtexts = [card(a, with_text=False) for a in qa]
        qv = self.emb.encode(qtexts)
        cand_i, cand_s = self.idx.search(qv, pool)
        pol_i, _ = self.pol_idx.search(self.emb.encode(qtexts, query=True), k_policy)
        # kNN-RAG neighbours (standardised features, exact)
        Xq = np.hstack(self.flat.transform(qdf)).astype("float32")
        d2 = (Xq ** 2).sum(1)[:, None] + self.Xf_sq[None, :] - 2 * Xq @ self.Xf.T
        part = np.argpartition(d2, k_cases - 1, axis=1)[:, :k_cases]
        knn_i = np.take_along_axis(part, np.argsort(np.take_along_axis(d2, part, 1), 1), 1)
        bundles = []
        for j, (cid, a) in enumerate(zip(qdf.case_id.values, qa)):
            cs = self.calf.loc[cid]
            qz = {k: np.array([logit(cs[f"e_{k}"]), logit(cs[f"p0_{k}"])]) for k in TREATMENTS}
            ev = {k: self._evidence(cand_i[j], cand_s[j], qz, k, m, qv[j]) for k in TREATMENTS}
            bundles.append(dict(
                case_id=int(cid), a=a, qtext=qtexts[j], evidence=ev,
                policy={k: self.policy_for(k) for k in TREATMENTS},
                calf={k: dict(tau=float(cs[f"tau_{k}"]), tau_sd=float(cs[f"tau_sd_{k}"]), e=float(cs[f"e_{k}"]),
                              p0=float(cs[f"p0_{k}"]), p1=float(cs[f"p1_{k}"])) for k in TREATMENTS},
                vanilla=dict(ids=cand_i[j][:k_cases].tolist(),
                             cases=[self._card_of(i) for i in cand_i[j][:k_cases]],
                             policy=[dict(publisher=self.pol.publisher[i], title=self.pol.title[i],
                                          text=self.pol.text[i], chunk_id=self.pol.chunk_id[i]) for i in pol_i[j]]),
                knn=dict(ids=knn_i[j].tolist(), cases=[self._card_of(i) for i in knn_i[j]])))
        return bundles
