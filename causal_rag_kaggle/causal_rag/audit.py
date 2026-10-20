"""
Synthesis & Audit Agent - emits the final decision with an auditable causal DAG instead of a wall of text.

Decision rule (fully deterministic given the inputs; the LLM only supplies the margin and a rationale):
    base      = DECLINE if margin >= thr else APPROVE
    counted audit flags (each is computed from the counterfactual + evidence tests, see counterfactual.py):
        llm_calf_disagree      LLM's response to a regime flip has the opposite sign to CALF's effect
        evidence_model_conflict matched-precedent effect and CALF effect have opposite signs (CI excludes 0)
        thin_overlap           matched evidence thin / unbalanced, or propensity outside [0.05, 0.95]
        evidence_ignored       LLM moved the wrong way when the matched evidence was reversed
        spurious_sensitive     decision changes when only policy-irrelevant fields change
    informational (not counted): policy_unsupported  (an applicable clause has no policy text in the corpus)
    decision  = REFER_TO_HUMAN  if  n_flags >= 2  or (n_flags >= 1 and |margin - thr| < band)   else base
"""
import base64
import html
import io
import json
import numpy as np
import pandas as pd

from . import causal_graph as cgm
from .config import TREATMENTS, RAW_FIELDS, OUTCOME
from .prompts import active_flags, SYSTEM_BUILDERS, RATIONALE_INSTR

MATERIAL = 0.01          # |tau| >= 1 percentage point counts as a material effect
RESPONSE = 0.005         # LLM P(decline) change below this is "no response"
PROHIBITED = ["race", "color", "religion", "national_origin", "sex", "gender", "marital_status",
              "age", "public_assistance"]


def sig(x):
    return 1 / (1 + np.exp(-x))


def audit_case(b, cf, margin, thr, band):
    """b: bundle; cf: DataFrame rows of the counterfactual agent for THIS case and system causal_rag."""
    a, flags, clauses = b["a"], [], []
    act = active_flags(a)
    for tk, on in act.items():
        if not on:
            continue
        ev, cs, pol = b["evidence"][tk], b["calf"][tk], b["policy"][tk]
        rf = cf[(cf.kind == "regime_flip") & (cf.clause == tk)]
        rv = cf[(cf.kind == "reverse") & (cf.clause == tk)]
        row = dict(clause=tk, label=TREATMENTS[tk]["clause"], tau=cs["tau"], tau_sd=cs["tau_sd"],
                   local_diff=ev["diff"], local_ci=[ev["lo"], ev["hi"]], matched=[ev["n_t"], ev["n_c"]],
                   policy_supported=pol["supported"],
                   policy_chunk=pol["chunks"][0]["chunk_id"] if pol["chunks"] else None,
                   llm_dP=None, agrees=None)
        if len(rf):
            r = rf.iloc[0]
            row["llm_dP"] = float(r.dP * r.dir)                 # in the direction "clause removed" -> sign flipped by dir
            if abs(cs["tau"]) >= MATERIAL and abs(r.dP) >= RESPONSE:
                row["agrees"] = bool(np.sign(r.dP * r.dir) == np.sign(cs["tau"]))
                if not row["agrees"]:
                    flags.append(f"llm_calf_disagree:{tk}")
        if (abs(ev["diff"]) >= 0.02 and abs(cs["tau"]) >= 0.02 and np.sign(ev["diff"]) != np.sign(cs["tau"])
                and (ev["lo"] > 0 or ev["hi"] < 0)):
            flags.append(f"evidence_model_conflict:{tk}")
        if ev["thin"] or not (0.05 < cs["e"] < 0.95):
            flags.append(f"thin_overlap:{tk}")
        if len(rv) and abs(ev["diff"]) >= 0.03 and abs(rv.iloc[0].dP) >= RESPONSE:
            if np.sign(rv.iloc[0].dP) == np.sign(ev["diff"]):   # reversed evidence should move OPPOSITE to original diff
                flags.append(f"evidence_ignored:{tk}")
        clauses.append(row)
    sp = cf[cf.kind == "spurious"]
    if len(sp):
        s = sp.iloc[0]
        if (s.base_margin >= thr) != (s.new_margin >= thr) or abs(s.dP) > 0.05:
            flags.append("spurious_sensitive")
    info = [f"policy_unsupported:{c['clause']}" for c in clauses if not c["policy_supported"]]
    base = "DECLINE" if margin >= thr else "APPROVE"
    n = len(flags)
    decision = "REFER_TO_HUMAN" if (n >= 2 or (n >= 1 and abs(margin - thr) < band)) else base
    return dict(case_id=b["case_id"], decision=decision, base_decision=base, margin=float(margin),
                p_decline_llm=float(sig(margin)), threshold=float(thr), band=float(band),
                audit_status="pass" if n == 0 else "review", flags=flags, informational=info, clauses=clauses,
                regulatory_input_check=dict(model_inputs=RAW_FIELDS, prohibited_bases_present=[
                    f for f in RAW_FIELDS if f.lower() in PROHIBITED],
                    note="credit_age_yrs is length of credit history, not applicant age; addr_state / emp_title / title "
                         "are policy-irrelevant and tested for sensitivity (spurious_sensitive)."),
                evidence_case_ids={tk: dict(treated=b["evidence"][tk]["ids_t"], control=b["evidence"][tk]["ids_c"])
                                   for tk in TREATMENTS if act[tk]})


def render_dag(g, rec, path, title=None):
    G, U = cgm.to_networkx(g)
    hl = [TREATMENTS[c["clause"]]["var"] for c in rec["clauses"]]
    lab, col = {}, {}
    for c in rec["clauses"]:
        v = TREATMENTS[c["clause"]]["var"]
        if (v, OUTCOME) in G.edges:
            ag = "?" if c["agrees"] is None else ("LLM agrees" if c["agrees"] else "LLM DISAGREES")
            lab[(v, OUTCOME)] = f"tau {100 * c['tau']:+.1f}pp\n{ag}"
            col[(v, OUTCOME)] = "#64748b" if c["agrees"] is None else ("#16a34a" if c["agrees"] else "#dc2626")
    cgm.draw_graph((G, U), path, highlight=hl, edge_labels=lab, edge_colors=col, figsize=(14, 8),
                   title=title or f"case {rec['case_id']}: {rec['decision']} | audit {rec['audit_status']}")


def html_report(rec, png_path, rationale=""):
    img = base64.b64encode(open(png_path, "rb").read()).decode()
    rows = "".join(
        f"<tr><td>{html.escape(c['label'])}</td><td>{100 * c['tau']:+.1f} pp</td><td>{100 * c['local_diff']:+.1f} pp "
        f"[{100 * c['local_ci'][0]:+.1f}, {100 * c['local_ci'][1]:+.1f}] (n={c['matched']})</td>"
        f"<td>{'yes' if c['policy_supported'] else 'NO - analyst-defined'}</td><td>{c['agrees']}</td></tr>"
        for c in rec["clauses"])
    return f"""<html><body style="font-family:sans-serif;max-width:1000px;margin:auto">
<h2>Case {rec['case_id']} - {rec['decision']} (audit: {rec['audit_status']})</h2>
<p>LLM P(decline) = {rec['p_decline_llm']:.2f}; threshold margin {rec['threshold']:.2f}; flags: {', '.join(rec['flags']) or 'none'}</p>
<table border=1 cellpadding=5><tr><th>Clause</th><th>CALF effect</th><th>Matched precedent</th><th>Policy text</th><th>LLM agrees</th></tr>{rows}</table>
<h3>Causal DAG</h3><img style="width:100%" src="data:image/png;base64,{img}"/>
<h3>Rationale (generated text - the table above is the audit record)</h3><p>{html.escape(rationale)}</p>
<h3>Record</h3><pre>{html.escape(json.dumps(rec, indent=1, default=str))}</pre></body></html>"""


def rationale(llm, b, rec, system="causal_rag"):
    msgs = SYSTEM_BUILDERS[system](b["a"], b)
    msgs[-1] = dict(msgs[-1], content=msgs[-1]["content"].split("\nShould this loan")[0] +
                    f"\n\nFinal decision: {rec['decision']}. Audit flags: {', '.join(rec['flags']) or 'none'}." + RATIONALE_INSTR)
    return llm.generate(msgs, max_new_tokens=160)
