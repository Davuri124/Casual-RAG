"""Applicant cards (real loan -> text) and prompt builders for the four compared systems."""
import math
import numpy as np

from .config import TREATMENTS

SYSTEM = ("You are a senior consumer-credit underwriting analyst. You decide whether a 36-month unsecured "
          "personal loan should be approved. Base the decision on the information provided. "
          "Answer with a single letter.")
QUESTION = ("\nShould this loan be approved?\nA = Approve (expected to be repaid in full)\n"
            "B = Decline (expected to default)\nAnswer with one letter only.")


def _num(x, nd=1):
    return "unknown" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def card(a, outcome=None, with_text=True):
    """Deterministic text rendering of a loan application (dict of RAW_FIELDS). Perturbation = edit dict, re-render."""
    lti = a["loan_amnt"] / a["annual_inc"]
    s = (f"Applicant: annual income ${a['annual_inc']:,.0f}; employed {_num(a['emp_years'], 0)} years; "
         f"home ownership {a['home_ownership']}; state {a['addr_state']}; income verification: {a['verification_status']}.\n"
         f"Credit profile: FICO {a['fico']:.0f}; debt-to-income {a['dti']:.1f}%; revolving utilization {a['revol_util']:.0f}%; "
         f"revolving balance ${a['revol_bal']:,.0f}; delinquencies (2y) {a['delinq_2yrs']:.0f}; inquiries (6m) {a['inq_last_6mths']:.0f}; "
         f"open accounts {a['open_acc']:.0f}; total accounts {a['total_acc']:.0f}; public records {a['pub_rec']:.0f}; "
         f"mortgage accounts {_num(a['mort_acc'], 0)}; credit history {a['credit_age_yrs']:.1f} years.\n"
         f"Request: ${a['loan_amnt']:,.0f} for {str(a['purpose']).replace('_', ' ')} (loan-to-income {100 * lti:.1f}%).")
    if with_text and (a.get("emp_title") or a.get("title")):
        s += f"\nEmployer title: {a.get('emp_title') or 'n/a'}; loan title: {a.get('title') or 'n/a'}."
    if outcome is not None:
        s += f"\nOUTCOME: {'DEFAULTED (charged off)' if outcome else 'repaid in full'}."
    return s


def active_flags(a):
    class _R:  # tiny adapter so config lambdas (d.dti > ...) work on a dict
        pass
    r = _R()
    r.dti, r.revol_util, r.fico = a["dti"], a["revol_util"], a["fico"]
    r.lti = a["loan_amnt"] / a["annual_inc"]
    return {k: bool(spec["define"](r)) for k, spec in TREATMENTS.items()}


def _pct(k, n):
    return f"{100 * k / max(n, 1):.0f}%"


def _excerpt(txt, n=70):
    w = txt.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def _wrap(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_norag(a, b=None):
    return _wrap(SYSTEM, "LOAN APPLICATION\n" + card(a) + "\n" + QUESTION)


def prompt_vanilla(a, b):
    pol = "\n".join(f"[{c['publisher']} - {c['title']}] {_excerpt(c['text'])}" for c in b["vanilla"]["policy"])
    cases = "\n\n".join(f"Case {i + 1}:\n{c}" for i, c in enumerate(b["vanilla"]["cases"]))
    return _wrap(SYSTEM, f"LOAN APPLICATION\n{card(a)}\n\nRELEVANT POLICY EXCERPTS\n{pol}\n\n"
                         f"MOST SIMILAR PAST LOANS (by text similarity)\n{cases}\n{QUESTION}")


def prompt_knn(a, b):
    cases = "\n\n".join(f"Case {i + 1}:\n{c}" for i, c in enumerate(b["knn"]["cases"]))
    return _wrap(SYSTEM, f"LOAN APPLICATION\n{card(a)}\n\nNEAREST PAST LOANS (by feature distance)\n{cases}\n{QUESTION}")


def prompt_causal(a, b, reverse=(), remove=(), show_calf=True):
    """
    Causal-RAG prompt. For every clause that APPLIES to the applicant it shows: policy basis, matched
    like-for-like precedent (treated vs control, same balance score) and the CALF effect estimate.
    `reverse`: clause keys whose matched-precedent counts are swapped (counterfactual evidence test).
    `remove` : clause keys whose precedent block is dropped (evidence-ablation test).
    """
    flags = active_flags(a)
    blocks, n = [], 0
    for k, on in flags.items():
        if not on:
            continue
        n += 1
        spec, ev, pol, cf = TREATMENTS[k], b["evidence"][k], b["policy"][k], b["calf"][k]
        t = f"Condition {n}: {spec['clause']} - applies to this applicant."
        if pol["supported"]:
            c = pol["chunks"][0]
            t += f"\n  Policy basis [{c['publisher']} - {c['title']}]: {_excerpt(c['text'], 60)}"
        elif pol.get("topical") and pol["chunks"]:
            c = pol["chunks"][0]
            t += (f"\n  Policy context (topic only; the policy text does not state this cutoff, so the threshold is "
                  f"analyst-defined) [{c['publisher']} - {c['title']}]: {_excerpt(c['text'], 40)}")
        else:
            t += "\n  Policy basis: no policy text in the corpus covers this condition (analyst-defined risk factor)."
        if k not in remove:
            yt, nt, yc, nc, d, lo, hi = ev["y_t"], ev["n_t"], ev["y_c"], ev["n_c"], ev["diff"], ev["lo"], ev["hi"]
            if k in reverse:
                yt, nt, yc, nc, d, lo, hi = yc, nc, yt, nt, -d, -hi, -lo
            t += (f"\n  Matched precedent (past borrowers compared like-for-like on background factors): "
                  f"WITH the condition {yt} of {nt} defaulted ({_pct(yt, nt)}); WITHOUT it {yc} of {nc} defaulted "
                  f"({_pct(yc, nc)}). Difference {100 * d:+.1f} pp (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})."
                  + (" [thin overlap - interpret with caution]" if ev["thin"] else ""))
        if show_calf:
            t += (f"\n  Estimated causal effect of this condition on THIS applicant's default probability: "
                  f"{100 * cf['tau']:+.1f} pp (model spread +/-{100 * cf['tau_sd']:.1f}).")
        blocks.append(t)
    body = "\n\n".join(blocks) if blocks else "No underwriting condition is triggered for this applicant."
    guide = ("Causal evidence is adjusted for confounding background factors: judge conditions by their estimated "
             "effect, not by how often they co-occur with default.")
    return _wrap(SYSTEM, f"LOAN APPLICATION\n{card(a)}\n\nAUDITED UNDERWRITING CONDITIONS\n{body}\n\n{guide}\n{QUESTION}")


def clause_value(a, k):
    return {"dti_gt25": f"DTI {a['dti']:.1f}%", "util_gt75": f"revolving utilization {a['revol_util']:.0f}%",
            "fico_lt680": f"FICO {a['fico']:.0f}",
            "lti_gt20": f"loan-to-income {100 * a['loan_amnt'] / a['annual_inc']:.1f}%"}[k]


def _policy_line(pol):
    if pol["supported"]:
        c = pol["chunks"][0]
        return f"\n  Policy basis [{c['publisher']} - {c['title']}]: {_excerpt(c['text'], 60)}"
    if pol.get("topical") and pol["chunks"]:
        c = pol["chunks"][0]
        return (f"\n  Policy context (topic only; the policy text does not state this cutoff, so the threshold is "
                f"analyst-defined) [{c['publisher']} - {c['title']}]: {_excerpt(c['text'], 40)}")
    return "\n  Policy basis: no policy text in the corpus covers this condition (analyst-defined risk factor)."


def prompt_causal_smooth(a, b, reverse=(), remove=(), show_calf=True):
    """
    v1.2 'smooth' Causal-RAG prompt. Fixes the cliff seen in v1.0/v1.1 (a condition block appeared/disappeared when a
    field crossed its cutoff): ALL four conditions are always shown in an identical layout with the applicant's actual
    value and a TRIGGERED / not triggered status, so crossing a cutoff changes one word, not the structure.
    """
    flags, blocks = active_flags(a), []
    for n, (k, spec) in enumerate(TREATMENTS.items(), 1):
        ev, pol, cf = b["evidence"][k], b["policy"][k], b["calf"][k]
        cmp_ = ">" if spec["side"] == "gt" else "<"
        t = (f"Condition {n}: {spec['clause']}. Applicant: {clause_value(a, k)} -> "
             f"{'TRIGGERED' if flags[k] else 'not triggered'} (cutoff {cmp_} {spec['thr']:g}).")
        t += _policy_line(pol)
        if k not in remove:
            yt, nt, yc, nc, d, lo, hi = ev["y_t"], ev["n_t"], ev["y_c"], ev["n_c"], ev["diff"], ev["lo"], ev["hi"]
            if k in reverse:
                yt, nt, yc, nc, d, lo, hi = yc, nc, yt, nt, -d, -hi, -lo
            t += (f"\n  Matched precedent (past borrowers compared like-for-like on background factors): inside this regime "
                  f"{yt} of {nt} defaulted ({_pct(yt, nt)}); outside it {yc} of {nc} defaulted ({_pct(yc, nc)}). "
                  f"Difference {100 * d:+.1f} pp (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})."
                  + (" [thin overlap - interpret with caution]" if ev["thin"] else ""))
        if show_calf:
            t += (f"\n  Estimated effect on THIS applicant's default probability of being inside this regime rather "
                  f"than outside it: {100 * cf['tau']:+.1f} pp (model spread +/-{100 * cf['tau_sd']:.1f}).")
        blocks.append(t)
    guide = ("The cutoffs are analyst-defined conventions, not cliffs: an applicant just inside and just outside a cutoff "
             "has almost the same risk. Weigh the estimated effects together with the whole profile.")
    return _wrap(SYSTEM, f"LOAN APPLICATION\n{card(a)}\n\nAUDITED UNDERWRITING CONDITIONS\n" + "\n\n".join(blocks) +
                 f"\n\n{guide}\n{QUESTION}")


UNIT = {"dti_gt25": "points", "util_gt75": "points", "fico_lt680": "points", "lti_gt20": "points of income"}


def _dist_text(a, k):
    spec = TREATMENTS[k]
    v = {"dti_gt25": a["dti"], "util_gt75": a["revol_util"], "fico_lt680": a["fico"],
         "lti_gt20": 100 * a["loan_amnt"] / a["annual_inc"]}[k]
    thr = spec["thr"] * (100 if k == "lti_gt20" else 1)
    d = v - thr
    return f"{abs(d):.1f} {UNIT[k]} {'above' if d > 0 else 'below'} the cutoff"


def prompt_conditions_only(a, b=None):
    """CONTROL: tells the LLM which standard risk flags apply - no policy text, no precedent, no effect numbers.
    Tests whether Causal-RAG's gain is just 'the conditions were flagged' (feature-engineering effect)."""
    flags = active_flags(a)
    on = [TREATMENTS[k]["clause"] for k, f in flags.items() if f]
    body = "\n".join(f"- {c}" for c in on) if on else "None of the standard risk flags is triggered."
    return _wrap(SYSTEM, f"LOAN APPLICATION\n{card(a)}\n\nSTANDARD RISK FLAGS TRIGGERED\n{body}\n{QUESTION}")


def prompt_graded(a, b, reverse=(), remove=(), show_calf=True):
    """
    v1.3 'graded' Causal-RAG prompt: like the smooth prompt but WITHOUT any TRIGGERED / not-triggered word. Each condition
    shows the applicant's value and its signed distance to the cutoff, so crossing a cutoff changes a number, not a label.
    """
    blocks = []
    for n, (k, spec) in enumerate(TREATMENTS.items(), 1):
        ev, pol, cf = b["evidence"][k], b["policy"][k], b["calf"][k]
        cmp_ = ">" if spec["side"] == "gt" else "<"
        t = (f"Condition {n}: {spec['clause']}. Applicant: {clause_value(a, k)} - {_dist_text(a, k)} "
             f"(regime boundary {cmp_} {spec['thr'] * (100 if k == 'lti_gt20' else 1):g}).")
        t += _policy_line(pol)
        if k not in remove:
            yt, nt, yc, nc, d, lo, hi = ev["y_t"], ev["n_t"], ev["y_c"], ev["n_c"], ev["diff"], ev["lo"], ev["hi"]
            if k in reverse:
                yt, nt, yc, nc, d, lo, hi = yc, nc, yt, nt, -d, -hi, -lo
            t += (f"\n  Matched precedent (past borrowers compared like-for-like on background factors): inside this regime "
                  f"{yt} of {nt} defaulted ({_pct(yt, nt)}); outside it {yc} of {nc} defaulted ({_pct(yc, nc)}). "
                  f"Difference {100 * d:+.1f} pp (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})."
                  + (" [thin overlap - interpret with caution]" if ev["thin"] else ""))
        if show_calf:
            t += (f"\n  Estimated effect on THIS applicant's default probability of being inside this regime rather "
                  f"than outside it: {100 * cf['tau']:+.1f} pp (model spread +/-{100 * cf['tau_sd']:.1f}).")
        blocks.append(t)
    guide = ("The cutoffs are analyst-defined conventions, not cliffs: an applicant just inside and just outside a cutoff "
             "has almost the same risk. Weigh the estimated effects together with the whole profile.")
    return _wrap(SYSTEM, f"LOAN APPLICATION\n{card(a)}\n\nAUDITED UNDERWRITING CONDITIONS\n" + "\n\n".join(blocks) +
                 f"\n\n{guide}\n{QUESTION}")


# ---------------------------------------------------------------- v2.0: band-table prompts (no cutoff language)
BAND_EDGES = {   # field -> (display name, edges, labels)
    "dti": ("DTI", [-1, 10, 15, 20, 25, 30, 35, 1000], ["<10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-35%", "35%+"]),
    "revol_util": ("Revolving utilization", [-1, 25, 50, 75, 90, 1000], ["<25%", "25-50%", "50-75%", "75-90%", "90%+"]),
    "fico": ("FICO", [0, 680, 700, 720, 740, 1000], ["<680", "680-699", "700-719", "720-739", "740+"]),
    "lti": ("Loan-to-income", [-1, 0.10, 0.15, 0.20, 0.25, 1000], ["<10%", "10-15%", "15-20%", "20-25%", "25%+"]),
}
BAND_TABLES = {}


def compute_band_tables(idx_df):
    """UNADJUSTED historical default rate by value band of each clause field, from labelled historical loans only."""
    import pandas as pd
    d = idx_df.copy()
    d["lti"] = d.loan_amnt / d.annual_inc
    out = {}
    for f, (name, edges, labels) in BAND_EDGES.items():
        b = pd.cut(d[f], bins=edges, labels=labels, right=False)
        g = d.groupby(b, observed=False).default.agg(["mean", "size"])
        out[f] = [(lab, float(g.loc[lab, "mean"]), int(g.loc[lab, "size"])) for lab in labels]
    return out


def set_band_tables(tables):
    BAND_TABLES.clear()
    BAND_TABLES.update(tables)


def _band_block():
    if not BAND_TABLES:
        raise RuntimeError("Band tables not set: call prompts.set_band_tables(prompts.compute_band_tables(index_df)) first.")
    return "\n".join(f"- {BAND_EDGES[f][0]}: " + "; ".join(f"{lab} {100 * r:.1f}% (n={n:,})" for lab, r, n in rows)
                     for f, rows in BAND_TABLES.items())


BAND_TITLE = "HISTORICAL DEFAULT RATES BY BAND (all past 36-month loans 2012-2014, unadjusted for other factors)"


def prompt_bands_only(a, b=None):
    """CONTROL / smoothness probe: applicant + unadjusted default-rate-by-band tables. No cutoffs, no flags, no regime wording,
    no precedent, no CALF numbers: any jump at a former cutoff can only come from the model itself."""
    return _wrap(SYSTEM, f"LOAN APPLICATION\n{card(a)}\n\n{BAND_TITLE}\n{_band_block()}\n{QUESTION}")


def prompt_graded_bands(a, b, reverse=(), remove=(), show_calf=True):
    """Graded Causal-RAG prompt PLUS the band tables (a smooth reference of how risk changes with each value)."""
    msgs = prompt_graded(a, b, reverse=reverse, remove=remove, show_calf=show_calf)
    u = msgs[-1]["content"]
    marker = "\n\nThe cutoffs are analyst-defined"
    msgs[-1] = dict(msgs[-1], content=u.replace(marker, f"\n\n{BAND_TITLE}\n{_band_block()}{marker}", 1))
    return msgs


def prompt_calf_only(a, b):                 # ablation: CALF effect numbers, NO matched precedent
    return prompt_causal(a, b, remove=tuple(TREATMENTS))


def prompt_precedent_only(a, b):            # ablation: matched precedent, NO CALF effect numbers
    return prompt_causal(a, b, show_calf=False)


SYSTEM_BUILDERS = {"no_rag": prompt_norag, "vanilla_rag": prompt_vanilla, "knn_rag": prompt_knn,
                   "causal_rag": prompt_causal, "causal_rag_smooth": prompt_causal_smooth,
                   "causal_calf_only": prompt_calf_only, "causal_precedent_only": prompt_precedent_only,
                   "causal_conditions_only": prompt_conditions_only, "causal_graded": prompt_graded,
                   "bands_only": prompt_bands_only, "causal_graded_bands": prompt_graded_bands}
CAUSAL_SYSTEMS = ("causal_rag", "causal_rag_smooth", "causal_graded", "causal_graded_bands")

RATIONALE_INSTR = ("\nNow write a 3-sentence justification for your decision. Cite only the conditions listed "
                   "above by number. Do not introduce facts that are not in the application or the evidence.")
