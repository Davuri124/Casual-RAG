"""
Live demo: pick a REAL 2015 Lending Club loan (or edit any field), run the full Causal-RAG audit in ~15-30 s, and watch how
different prompt designs react as you slide one value across its cutoff.

  LiveAuditor : engine (CALF effects -> retrieval -> LLM scoring -> counterfactual agent -> audit record + DAG)
  build_ui    : Gradio Blocks app (Audit tab + What-if sweep tab)

The sweep tab is the most informative: it plots P(decline) against a continuous field (DTI, utilization, FICO,
loan-to-income) for several systems with the retrieved evidence held fixed. A system that reasons smoothly gives a ramp;
a system that reacts to a label gives a step at the cutoff (the 'cliff' measured in the v1.1 / v1.2 counterfactual tests).
"""
import tempfile
import numpy as np
import pandas as pd

from . import audit as au
from .config import TREATMENTS, RAW_FIELDS, add_features
from .counterfactual import CounterfactualAgent, _set_clause_value
from .prompts import SYSTEM_BUILDERS

SWEEP = {
    "DTI (%)": ("dti_gt25", np.linspace(5, 45, 9)),
    "Revolving utilization (%)": ("util_gt75", np.linspace(5, 100, 9)),
    "FICO score": ("fico_lt680", np.linspace(620, 800, 10)),
    "Loan-to-income (ratio)": ("lti_gt20", np.linspace(0.05, 0.45, 9)),
}
SHOW = ["no_rag", "causal_conditions_only", "causal_rag_smooth", "causal_graded", "causal_graded_bands"]
LABELS = {"no_rag": "No RAG (plain LLM)", "causal_conditions_only": "Flags only (control)",
          "causal_rag_smooth": "Causal-RAG smooth", "causal_graded": "Causal-RAG graded",
          "causal_graded_bands": "Causal-RAG graded + bands"}
CUTOFF = {"dti_gt25": 25.0, "util_gt75": 75.0, "fico_lt680": 680.0, "lti_gt20": 0.20}


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


class LiveAuditor:
    def __init__(self, llm, retriever, calf_scorer, graph, test_df, medians, thr=None, band=None, system="causal_graded"):
        self.llm, self.r, self.calf, self.graph = llm, retriever, calf_scorer, graph
        from . import prompts as _pr
        if not _pr.BAND_TABLES:
            _pr.set_band_tables(_pr.compute_band_tables(retriever.df))      # needed by the band prompts
        self.test, self.medians, self.system = test_df.reset_index(drop=True), medians, system
        self.thr, self.band = thr, band
        self.agent = CounterfactualAgent(llm, retriever.df, systems=(system,), boundary_systems=(system,))
        self._cache = {}
        self.by_id = self.test.set_index("case_id")

    # ------------------------------------------------------------------ helpers
    def case_ids(self, n=300):
        return self.test.case_id.values[:n].tolist()

    def applicant(self, case_id):
        row = self.by_id.loc[int(case_id)]
        a = {f: row[f] for f in RAW_FIELDS}
        for f in ("emp_years", "mort_acc"):
            if row.get(f + "_na", 0) == 1:
                a[f] = float("nan")
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in a.items()}

    def _key(self, a):
        return tuple((k, round(v, 4) if isinstance(v, float) else v) for k, v in sorted(a.items()))

    def _bundle(self, a):
        k = self._key(a)
        if k in self._cache:
            return self._cache[k]
        sc = self.calf.score(a)
        self.r.calf.loc[-1] = {f"{c}_{tk}": sc[tk][c] for tk in TREATMENTS for c in ("e", "p0", "p1", "tau", "tau_sd")}
        q = add_features(pd.DataFrame([{**a, "case_id": -1}]), self.medians)
        b = self.r.build_bundles(q)[0]
        if len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[k] = b
        return b

    def calibrate(self, n=40, seed=0):
        """Label-free operating point when no saved scores are available: decline the riskiest 20% of n real test loans."""
        sub = self.test.sample(n, random_state=seed)
        bs = self.r.build_bundles(sub)
        m = self.llm.margins([SYSTEM_BUILDERS[self.system](b["a"], b) for b in bs], progress=False)
        self.thr, self.band = float(np.quantile(m, .8)), float(.15 * m.std())
        return self.thr, self.band

    # ------------------------------------------------------------------ audit
    def analyze(self, a):
        if self.thr is None:
            self.calibrate()
        b = self._bundle(a)
        systems = list(dict.fromkeys(SHOW + [self.system]))
        m = self.llm.margins([SYSTEM_BUILDERS[s](a, b) for s in systems], progress=False)
        margins = dict(zip(systems, [float(x) for x in m]))
        cf = self.agent.run([b], {self.system: np.array([margins[self.system]])})
        rec = au.audit_case(b, cf[cf.system == self.system], margins[self.system], self.thr, self.band)
        png = tempfile.mktemp(suffix=".png")
        au.render_dag(self.graph, rec, png)
        return dict(rec=rec, bundle=b, margins=margins, png=png)

    def sweep(self, a, label, systems=SHOW):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tk, vals = SWEEP[label]
        b = self._bundle(a)
        prompts = [SYSTEM_BUILDERS[s](_set_clause_value(a, tk, v), b) for s in systems for v in vals]
        p = sig(self.llm.margins(prompts, progress=False)).reshape(len(systems), len(vals))
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for s, row in zip(systems, p):
            ax.plot(vals, row, marker="o", label=LABELS.get(s, s))
        ax.axvline(CUTOFF[tk], color="k", ls="--", lw=1)
        ax.text(CUTOFF[tk], 1.02, " cutoff", fontsize=8)
        ax.set_ylim(0, 1.08); ax.set_xlabel(label); ax.set_ylabel("LLM P(decline)")
        ax.set_title("What-if sweep (retrieved evidence held fixed): ramp = smooth reasoning, step = reacts to the cutoff")
        ax.title.set_fontsize(8.5); ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.tight_layout()
        tab = pd.DataFrame({"value": np.round(vals, 3), **{LABELS.get(s, s): np.round(r, 3) for s, r in zip(systems, p)}})
        return fig, tab


def build_ui(aud: LiveAuditor):
    import gradio as gr
    ids = aud.case_ids()
    a0 = aud.applicant(ids[0])

    def collect(inc, dti, util, fico, amt, base_id):
        a = aud.applicant(base_id)
        a.update(annual_inc=float(inc), dti=float(dti), revol_util=float(util), fico=float(fico), loan_amnt=float(amt))
        return a

    def load(cid):
        a = aud.applicant(cid)
        return a["annual_inc"], a["dti"], a["revol_util"], a["fico"], a["loan_amnt"], card_text(a)

    def card_text(a):
        from .prompts import card
        return card(a)

    def run_audit(cid, inc, dti, util, fico, amt, with_text):
        try:
            a = collect(inc, dti, util, fico, amt, cid)
            out = aud.analyze(a)
            rec = out["rec"]
            head = (f"{rec['decision']}   |   audit: {rec['audit_status']}   |   flags: {', '.join(rec['flags']) or 'none'}"
                    f"\nP(decline) by system: " + ", ".join(f"{LABELS.get(s, s)} {sig(m):.2f}" for s, m in out['margins'].items()))
            clauses = pd.DataFrame([{
                "condition": c["label"], "CALF effect (pp)": round(100 * c["tau"], 2),
                "matched precedent (pp)": round(100 * c["local_diff"], 2),
                "policy states cutoff": c["policy_supported"], "LLM agrees with CALF": c["agrees"]} for c in rec["clauses"]])
            if clauses.empty:
                clauses = pd.DataFrame([{"condition": "no condition triggered for this applicant"}])
            why = au.rationale(aud.llm, out["bundle"], rec, aud.system) if with_text else "(tick 'generate rationale' to produce text)"
            import json
            return head, clauses, out["png"], why, json.dumps(rec, indent=1, default=str)
        except Exception as e:                                             # show the problem in the UI instead of a stack trace
            return f"ERROR: {type(e).__name__}: {e}", pd.DataFrame(), None, "", ""

    def run_sweep(cid, inc, dti, util, fico, amt, label):
        try:
            fig, tab = aud.sweep(collect(inc, dti, util, fico, amt, cid), label)
            return fig, tab
        except Exception as e:
            return None, pd.DataFrame([{"error": f"{type(e).__name__}: {e}"}])

    with gr.Blocks(title="Causal-RAG live audit") as ui:
        gr.Markdown("## Causal-RAG: live counterfactual audit on real 2015 Lending Club loans\n"
                    "Pick a real loan, optionally edit values (what-if), then run the audit or the sweep. "
                    "First call after start is slower (model warm-up). Research demo, not a lending decision tool.")
        with gr.Row():
            cid = gr.Dropdown(ids, value=ids[0], label="real loan (case_id)")
            inc = gr.Number(value=a0["annual_inc"], label="annual income ($)")
            amt = gr.Number(value=a0["loan_amnt"], label="loan amount ($)")
        with gr.Row():
            dti = gr.Slider(0, 60, value=a0["dti"], step=0.5, label="DTI (%)")
            util = gr.Slider(0, 100, value=a0["revol_util"], step=1, label="revolving utilization (%)")
            fico = gr.Slider(600, 850, value=a0["fico"], step=1, label="FICO score")
        card_box = gr.Textbox(value=card_text(a0), label="application (real loan)", lines=6, interactive=False)
        controls = [cid, inc, dti, util, fico, amt]
        cid.change(load, cid, [inc, dti, util, fico, amt, card_box])
        with gr.Tab("Audit"):
            with gr.Row():
                why = gr.Checkbox(label="generate rationale text (slower)", value=False)
                btn = gr.Button("Run causal audit", variant="primary")
            head = gr.Textbox(label="decision", lines=2)
            clauses = gr.Dataframe(label="condition-by-condition audit")
            img = gr.Image(label="causal DAG (green = LLM agrees with CALF, red = disagrees)", type="filepath")
            rationale = gr.Textbox(label="rationale (generated text; the table is the audit record)", lines=4)
            js = gr.Code(label="full audit record", language="json")
            btn.click(run_audit, controls + [why], [head, clauses, img, rationale, js])
        with gr.Tab("What-if sweep"):
            lab = gr.Dropdown(list(SWEEP), value="DTI (%)", label="field to sweep")
            b2 = gr.Button("Run sweep (about 40 LLM calls)", variant="primary")
            plot = gr.Plot(label="P(decline) vs value")
            tab = gr.Dataframe(label="values")
            b2.click(run_sweep, controls + [lab], [plot, tab])
    return ui
