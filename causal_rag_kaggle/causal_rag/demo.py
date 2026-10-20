"""Gradio demo: pick a real 2015 loan, optionally edit fields (what-if), get decision + audit table + causal DAG."""
import json
import tempfile
import numpy as np
import pandas as pd

from . import audit as au
from .config import TREATMENTS, add_features, RAW_FIELDS
from .counterfactual import CounterfactualAgent
from .prompts import SYSTEM_BUILDERS
from .retrieval import row_to_a


def make_runner(llm, retriever, calf_scorer, graph, test_df, medians, thr, band, system="causal_rag"):
    agent = CounterfactualAgent(llm, retriever.df, systems=(system,), boundary_systems=(system,))
    by_id = test_df.set_index("case_id")

    def run(case_id, annual_inc, dti, revol_util, fico, loan_amnt, with_rationale):
        row = by_id.loc[int(case_id)]
        a = {f: row[f] for f in RAW_FIELDS}
        a.update(annual_inc=float(annual_inc), dti=float(dti), revol_util=float(revol_util), fico=float(fico),
                 loan_amnt=float(loan_amnt))
        sc = calf_scorer.score(a)
        retriever.calf.loc[-1] = {**{f"{c}_{tk}": sc[tk][c] for tk in TREATMENTS for c in ("e", "p0", "p1", "tau", "tau_sd")}}
        q = add_features(pd.DataFrame([{**a, "case_id": -1}]), medians)
        b = retriever.build_bundles(q)[0]
        m = llm.margins([SYSTEM_BUILDERS[system](a, b)], progress=False)
        cf = agent.run([b], {system: m})
        rec = au.audit_case(b, cf[cf.system == system], float(m[0]), thr, band)
        png = tempfile.mktemp(suffix=".png")
        au.render_dag(graph, rec, png)
        why = au.rationale(llm, b, rec, system) if with_rationale else ""
        table = pd.DataFrame([{"clause": c["label"], "CALF effect (pp)": round(100 * c["tau"], 2),
                               "matched precedent (pp)": round(100 * c["local_diff"], 2),
                               "policy text": c["policy_supported"], "LLM agrees": c["agrees"]} for c in rec["clauses"]])
        head = f"{rec['decision']}  |  audit: {rec['audit_status']}  |  flags: {', '.join(rec['flags']) or 'none'}"
        return head, table, png, why, json.dumps(rec, indent=1, default=str)
    return run


def launch(llm, retriever, calf_scorer, graph, test_df, medians, thr, band, share=True, system="causal_rag"):
    import gradio as gr
    run = make_runner(llm, retriever, calf_scorer, graph, test_df, medians, thr, band, system)
    ids = test_df.case_id.values[:300].tolist()
    with gr.Blocks(title="Causal-RAG underwriting audit") as ui:
        gr.Markdown("## Causal-RAG: counterfactual-audited underwriting (real 2015 Lending Club loans)")
        with gr.Row():
            cid = gr.Dropdown(ids, value=ids[0], label="case_id")
            inc = gr.Number(label="annual_inc", value=float(test_df.annual_inc.iloc[0]))
            dti = gr.Number(label="dti", value=float(test_df.dti.iloc[0]))
            util = gr.Number(label="revol_util", value=float(test_df.revol_util.iloc[0]))
            fico = gr.Number(label="fico", value=float(test_df.fico.iloc[0]))
            amt = gr.Number(label="loan_amnt", value=float(test_df.loan_amnt.iloc[0]))
        why = gr.Checkbox(label="generate rationale text", value=False)
        btn = gr.Button("Run audit")
        head = gr.Textbox(label="decision"); tab = gr.Dataframe(label="clause audit")
        img = gr.Image(label="causal DAG"); txt = gr.Textbox(label="rationale"); js = gr.Code(label="record", language="json")
        btn.click(run, [cid, inc, dti, util, fico, amt, why], [head, tab, img, txt, js])
        cid.change(lambda c: [float(test_df.set_index("case_id").loc[int(c)][f]) for f in
                              ("annual_inc", "dti", "revol_util", "fico", "loan_amnt")], cid, [inc, dti, util, fico, amt])
    ui.launch(share=share)
