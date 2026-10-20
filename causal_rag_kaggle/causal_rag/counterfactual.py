"""
Counterfactual Perturbation Agent - the causal sanity check on the LLM.

For every applicant and every clause k it edits the applicant (or the retrieved evidence), re-asks the same
LLM with the same prompt template, and compares the *direction* of the change with CALF's estimated effect.

Perturbation families
  regime_flip   do(T_k -> 1 - T_k): the clause field is moved to the MEDIAN value of the opposite arm of the matched,
                like-for-like precedent set (retrieval.py). This is the contrast CALF's tau_k actually estimates
                ("same background, other regime"), so the expected sign of the LLM's change is  dir * sign(tau_k).
  boundary_flip the same field moved just across the threshold (+-eps). Causally almost nothing changes, so a sound
                model should react only a little: this measures CLIFF SENSITIVITY (hard-rule behaviour), not consistency.
  reverse       (Causal-RAG only) matched-precedent counts of an active clause are swapped between arms.
                A model that really uses the evidence must move opposite to the original local effect.
  remove        (Causal-RAG only) the precedent block of an active clause is deleted.
  spurious      fields no clause references (state, employer title, loan title) are replaced by random real
                values from the index. The decision should NOT move (counted as a flip if it crosses the threshold).

The agent only needs `scorer.margins(list_of_chat_prompts) -> np.ndarray`, so any LLM (or a test double) plugs in.
"""
import copy
import numpy as np
import pandas as pd

from .config import TREATMENTS, SPURIOUS_FIELDS
from .prompts import SYSTEM_BUILDERS, active_flags, CAUSAL_SYSTEMS

FIELD = {"dti_gt25": "dti", "util_gt75": "revol_util", "fico_lt680": "fico"}
EPS = {"dti_gt25": 0.5, "util_gt75": 0.5, "fico_lt680": 1.0, "lti_gt20": 0.005}
LABEL = {"dti_gt25": "DTI", "util_gt75": "utilization", "fico_lt680": "FICO", "lti_gt20": "loan-to-income"}


def _set_clause_value(a, tk, v):
    a = copy.deepcopy(a)
    if tk == "lti_gt20":
        a["loan_amnt"] = float(np.clip(v * a["annual_inc"], 500, 40000))
    else:
        a[FIELD[tk]] = float(v)
    return a


def _cur(a, tk):
    return a["loan_amnt"] / a["annual_inc"] if tk == "lti_gt20" else a[FIELD[tk]]


def _target(tk, to_treated, v):
    """Clamp value v to the requested side of the threshold."""
    thr, side, e = TREATMENTS[tk]["thr"], TREATMENTS[tk]["side"], EPS[tk]
    want_high = (side == "gt") == to_treated          # treated&gt -> high ; control&lt -> high
    return max(v, thr + e) if want_high else min(v, thr - e)


def boundary_flip(a, tk):
    on = active_flags(a)[tk]
    thr, side, e = TREATMENTS[tk]["thr"], TREATMENTS[tk]["side"], EPS[tk]
    to_treated = not on
    want_high = (side == "gt") == to_treated
    new = thr + e if want_high else thr - e
    b = _set_clause_value(a, tk, new)
    if active_flags(b)[tk] == on:                      # could not cross (clipped) -> no valid counterfactual
        return None
    return b, (1 if to_treated else -1), f"{LABEL[tk]} {_cur(a, tk):.2f} -> {_cur(b, tk):.2f}"


def regime_flip(a, tk, bundle, idx_df):
    on = active_flags(a)[tk]
    ev = bundle["evidence"][tk]
    ids = ev["ids_c"] if on else ev["ids_t"]
    if not ids:
        return None
    sub = idx_df.iloc[ids]
    vals = (sub.loan_amnt / sub.annual_inc) if tk == "lti_gt20" else sub[FIELD[tk]]
    new = _target(tk, not on, float(np.median(vals)))
    b = _set_clause_value(a, tk, new)
    if active_flags(b)[tk] == on:
        return None
    return b, (-1 if on else 1), f"{LABEL[tk]} {_cur(a, tk):.2f} -> {_cur(b, tk):.2f}"


# ---------------------------------------------------------------- local cliff test (v1.4)
CLIFF_DELTA = {"dti_gt25": 1.0, "util_gt75": 2.0, "fico_lt680": 3.0, "lti_gt20": 0.01}


def cliff_points(tk):
    """Four values of the clause field, all other fields held at the applicant's own values:
       lo/hi = just below / above the cutoff (width 2d)  ->  the cross-cutoff contrast;
       p0/p1 = an equally wide interval well away from the cutoff (thr-5d .. thr-3d) -> the placebo contrast."""
    d, thr = CLIFF_DELTA[tk], TREATMENTS[tk]["thr"]
    return dict(lo=thr - d, hi=thr + d, p0=thr - 5 * d, p1=thr - 3 * d)


def spurious_edit(a, pools, rng):
    b = copy.deepcopy(a)
    for f in SPURIOUS_FIELDS:
        for _ in range(5):
            v = pools[f][rng.integers(len(pools[f]))]
            if v != a[f]:
                b[f] = v; break
    return b


def local_cliff_variants(a, tk):
    """Returns {name: edited applicant} or None if any point cannot be represented (e.g. loan amount clipped)."""
    out = {}
    for name, v in cliff_points(tk).items():
        b = _set_clause_value(a, tk, v)
        if abs(_cur(b, tk) - v) > 1e-6 * max(1.0, abs(v)):
            return None
        out[name] = b
    return out


class CounterfactualAgent:
    def __init__(self, scorer, idx_df, systems=("no_rag", "vanilla_rag", "causal_rag"),
                 boundary_systems=("causal_rag",), seed=0, boundary_only=()):
        self.sc, self.df, self.bonly = scorer, idx_df, tuple(boundary_only)
        self.systems, self.bsys, self.rng = systems, boundary_systems, np.random.default_rng(seed)
        self.pools = ({f: idx_df[f].astype(str)[idx_df[f].astype(str).str.len() > 0].values for f in SPURIOUS_FIELDS}
                      if idx_df is not None else {})

    def run(self, bundles, base):
        """base: dict system -> margins aligned with bundles. Returns long DataFrame of perturbation results."""
        jobs = []                                               # (i, system, kind, clause, dir, desc, prompt)
        for i, b in enumerate(bundles):
            a = b["a"]
            for sysname in self.systems:
                build = SYSTEM_BUILDERS[sysname]
                if sysname in self.bonly:                   # cliff test only (cheap): no regime / spurious / evidence jobs
                    for tk in TREATMENTS:
                        r = boundary_flip(a, tk)
                        if r:
                            jobs.append((i, sysname, "boundary_flip", tk, r[1], r[2], build(r[0], b)))
                    continue
                for tk in TREATMENTS:
                    r = regime_flip(a, tk, b, self.df)
                    if r:
                        jobs.append((i, sysname, "regime_flip", tk, r[1], r[2], build(r[0], b)))
                    if sysname in self.bsys:
                        r = boundary_flip(a, tk)
                        if r:
                            jobs.append((i, sysname, "boundary_flip", tk, r[1], r[2], build(r[0], b)))
                jobs.append((i, sysname, "spurious", "", 0, "state/employer/loan title",
                             build(spurious_edit(a, self.pools, self.rng), b)))
            for cs in [s for s in self.systems if s in CAUSAL_SYSTEMS]:
                for tk, on in active_flags(a).items():
                    if on:
                        jobs.append((i, cs, "reverse", tk, 0, "precedent counts swapped",
                                     SYSTEM_BUILDERS[cs](a, b, reverse=(tk,))))
                        jobs.append((i, cs, "remove", tk, 0, "precedent block removed",
                                     SYSTEM_BUILDERS[cs](a, b, remove=(tk,))))
        rows = []
        for sysname in self.systems:
            js = [j for j in jobs if j[1] == sysname]
            m = self.sc.margins([j[6] for j in js])
            for j, mj in zip(js, m):
                i, _, kind, tk, d, desc, _ = j
                b = bundles[i]
                m0 = base[sysname][i]
                rows.append(dict(
                    case_id=b["case_id"], system=sysname, kind=kind, clause=tk, dir=d, desc=desc,
                    tau=b["calf"][tk]["tau"] if tk else np.nan,
                    local_diff=b["evidence"][tk]["diff"] if tk else np.nan,
                    base_margin=float(m0), new_margin=float(mj),
                    dP=float(1 / (1 + np.exp(-mj)) - 1 / (1 + np.exp(-m0)))))
        return pd.DataFrame(rows)

    def run_local_cliff(self, bundles, systems):
        """
        Proper cliff test. Compares the change in P(decline) across the cutoff (thr-d -> thr+d) with the change over an
        equally wide interval far from it (thr-5d -> thr-3d). A system that reasons smoothly changes about as much in both
        (index ~ 1); a system reacting to a label/regime changes far more across the cutoff (index >> 1).
        Contrasts are signed so that positive = the riskier side gets the higher P(decline).
        """
        jobs, meta = [], []
        for i, b in enumerate(bundles):
            for tk in TREATMENTS:
                v = local_cliff_variants(b["a"], tk)
                if v is None:
                    continue
                for s in systems:
                    for name, a2 in v.items():
                        jobs.append(SYSTEM_BUILDERS[s](a2, b)); meta.append((b["case_id"], s, tk, name))
        m = self.sc.margins(jobs)
        p = 1 / (1 + np.exp(-m))
        df = pd.DataFrame(meta, columns=["case_id", "system", "clause", "point"]).assign(p=p)
        w = df.pivot_table(index=["case_id", "system", "clause"], columns="point", values="p").reset_index()
        sgn = np.where(w.clause.map(lambda k: TREATMENTS[k]["side"]) == "gt", 1.0, -1.0)
        w["cross"] = sgn * (w["hi"] - w["lo"])
        w["placebo"] = sgn * (w["p1"] - w["p0"])
        return w
