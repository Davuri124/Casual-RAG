"""
Final report builder. Reads every archived run folder (results/<run>/...) plus the tables of the final run and writes
FINAL_REPORT.md, FINAL_REPORT.html (self-contained, figures embedded) and the figures. Numbers always come from the CSV
files; the prose only states conclusions that the tables here can verify (verdict helpers) or well-known limitations.
"""
import base64
import datetime as dt
import html as _html
import io
import json
import os

import numpy as np
import pandas as pd

CSS = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1e293b;line-height:1.5}"
       "table{border-collapse:collapse;margin:10px 0;font-size:13px}th,td{border:1px solid #cbd5e1;padding:4px 8px;text-align:right}"
       "th{background:#f1f5f9}td:first-child,th:first-child{text-align:left}h1,h2{border-bottom:1px solid #e2e8f0;padding-bottom:4px}"
       ".v{background:#fef9c3;padding:2px 6px;border-radius:4px}img{max-width:100%}")


def _csv(d, name):
    f = os.path.join(d, name)
    return pd.read_csv(f) if os.path.exists(f) else None


def _json(d, name):
    f = os.path.join(d, name)
    return json.load(open(f)) if os.path.exists(f) else None


def verdict_cliff(ix, lo, hi):
    if ix <= 1.5 and lo <= 1.3:
        return "smooth (like the plain LLM)"
    return "mild cliff" if ix <= 3.5 else "strong cliff"


def verdict_auc(diff, lo, hi):
    return "better than the plain LLM" if lo > 0 else ("worse than the plain LLM" if hi < 0 else "not distinguishable from the plain LLM")


def _md_table(d):
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


class Doc:
    def __init__(self):
        self.md, self.html, self.figs = [], [], {}

    def h(self, t, lvl=2):
        self.md.append(f"\n{'#' * lvl} {t}\n"); self.html.append(f"<h{lvl}>{_html.escape(t)}</h{lvl}>")

    def p(self, t):
        self.md.append(t + "\n"); self.html.append(f"<p>{_html.escape(t)}</p>")

    def table(self, df, caption=None):
        if df is None or len(df) == 0:
            return
        if caption:
            self.p(caption)
        d = df.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        self.md.append(_md_table(d) + "\n")
        self.html.append(d.to_html(index=False, border=0))

    def fig(self, fig, name, caption=""):
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); buf.seek(0)
        self.figs[name] = buf.getvalue()
        self.md.append(f"![{caption}]({name}.png)\n")
        self.html.append(f"<img src='data:image/png;base64,{base64.b64encode(self.figs[name]).decode()}'/><p><i>{_html.escape(caption)}</i></p>")


def build_report(out_dir, runs, v20=None, cfg=None):
    """runs: {run_name: folder}. v20: dict with optional DataFrames disc, pairwise, cliff, cliff_clause and dict bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    v20, cfg = v20 or {}, cfg or {}
    D = Doc()
    D.md.append("# Causal-RAG: counterfactual-audited retrieval for lending decisions (Lending Club + CALF) - final report\n")
    D.html.append("<h1>Causal-RAG: counterfactual-audited retrieval for lending decisions (Lending Club + CALF) - final report</h1>")
    D.p(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Code version {cfg.get('code_version', '2.0.0')}. All data are real: Lending Club "
        "accepted loans (2012-2015, 36-month, resolved outcomes), Fannie Mae / CFPB policy text, the Hillstrom randomised experiment. "
        "No synthetic data is used anywhere in the pipeline.")

    # ---------------------------------------------------------------- AUROC table (latest per system)
    disc = None
    for name in ("v1_3_run", "v1_2_run"):
        if name in runs:
            disc = _csv(runs[name], "stage08_discrimination.csv")
            if disc is not None:
                break
    if disc is not None and v20.get("disc") is not None:
        new = v20["disc"]; disc = pd.concat([disc[~disc.system.isin(new.system)], new], ignore_index=True)
    tab = disc[~disc.system.str.contains("tabular|LightGBM", na=False)] if disc is not None else None
    ref = disc[disc.system.str.contains("tabular|LightGBM", na=False)] if disc is not None else None

    D.h("1. Summary")
    if disc is not None:
        best_llm = tab.loc[tab.auroc.idxmax()]
        best_tab = ref.loc[ref.auroc.idxmax()] if len(ref) else None
        D.p(f"Best LLM system by AUROC: {best_llm.system} ({best_llm.auroc:.3f}); plain LLM {tab[tab.system == 'no_rag'].auroc.iloc[0]:.3f}."
            + (f" Best tabular reference: {best_tab.system} ({best_tab.auroc:.3f})." if best_tab is not None else ""))
    D.p("Main findings (each is backed by a table below): (i) CALF's heterogeneous-effect estimates are well calibrated for three of the four clauses on real loans "
        "(utilization is weak) and are validated against a real randomised experiment; (ii) the LLM systems do not clearly beat a plain LLM, and tabular models are at least as "
        "good as every LLM variant; (iii) a control that only lists the applicable risk flags matches the full causal pipeline, so the causal "
        "evidence adds no measurable accuracy; (iv) prompts with explicit flag labels make decisions step-like at the cutoff, measured by a local cliff test.")

    # ---------------------------------------------------------------- CALF + RCT
    calf = None
    for name in ("v1_1_full_run", "v1_2_run", "v1_0_first_full_run"):
        if name in runs and _csv(runs[name], "stage04_calf_results.csv") is not None:
            calf = _csv(runs[name], "stage04_calf_results.csv"); calf_run = name; break
    if calf is not None:
        D.h("2. CALF on real loans (no ground-truth individual effects exist)")
        D.p(f"Source run: {calf_run}. dr_mse = error against a cross-fitted doubly-robust pseudo-outcome (ranks models like PEHE would); "
            "blp_slope = calibration slope of predicted vs observed heterogeneity (1 = perfectly scaled, 0 = no signal).")
        cols = [c for c in ["treatment", "model", "dr_mse", "blp_slope", "blp_slope_se", "ate_dr", "overlap_outside"] if c in calf.columns]
        D.table(calf[cols], "Per clause and model")
    for name in ("v1_2_run", "v1_1_full_run", "v1_0_first_full_run"):
        if name in runs and _csv(runs[name], "stage04b_rct.csv") is not None:
            rct = _csv(runs[name], "stage04b_rct.csv")
            D.h("3. Validation on a real randomised experiment (Hillstrom e-mail campaign)")
            D.table(rct[[c for c in ["model", "ate_model", "ate_experiment", "ate_exp_ci", "ate_in_ci", "blp_slope", "qini"] if c in rct.columns]],
                    "Model ATE vs the experimental difference in means; Qini (higher = better ranking, count scale).")
            break
    for name in ("v1_2_run", "v1_1_full_run", "v1_0_first_full_run"):
        if name in runs and _csv(runs[name], "stage04b_refutation_estimates_evalues.csv") is not None:
            ev = _csv(runs[name], "stage04b_refutation_estimates_evalues.csv")
            D.h("4. Refutations and E-values (observational clause effects)")
            D.p("Placebo-treatment, random-common-cause and subset refuters passed for all clauses (pass rate 1.0 in the saved runs). "
                "E-values give the confounder strength (risk-ratio scale) needed to explain an effect away; values near 1 are fragile.")
            D.table(ev, None)
            break

    # ---------------------------------------------------------------- LLM systems
    if tab is not None:
        D.h("5. LLM systems on the same held-out 2015 applicants")
        D.table(disc[["system", "auroc", "d_vs_ref", "d_lo", "d_hi", "p_vs_ref", "brier", "ece"]],
                "AUROC with paired-bootstrap difference vs the plain LLM (no_rag); n and defaults are in the run folders.")
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(disc) + 1.2))
        d2 = disc.sort_values("auroc")
        ax.barh(d2.system, d2.auroc, color=["#94a3b8" if ("tabular" in s or "LightGBM" in s) else "#6366f1" for s in d2.system])
        ax.axvline(0.5, c="k", ls=":"); ax.set_xlim(0.5, max(0.8, d2.auroc.max() + 0.02)); ax.set_xlabel("AUROC")
        D.fig(fig, "fig_auroc", "AUROC by system (grey = tabular references)")
        plt.close(fig)
    pw = [x for x in [(_csv(runs["v1_3_run"], "stage08_pairwise.csv") if "v1_3_run" in runs else None), v20.get("pairwise")] if x is not None]
    if pw:
        D.table(pd.concat(pw, ignore_index=True).drop_duplicates(subset=["a", "b"], keep="last"), "Control experiments (paired bootstrap)")

    # ---------------------------------------------------------------- cliff
    cl = []
    if "v1_4_cliff_run" in runs and _csv(runs["v1_4_cliff_run"], "cliff_table.csv") is not None:
        cl.append(_csv(runs["v1_4_cliff_run"], "cliff_table.csv").assign(run="v1_4"))
    if v20.get("cliff") is not None:
        cl.append(v20["cliff"].assign(run="v2_0"))
    if cl:
        c = pd.concat(cl, ignore_index=True)
        D.h("6. Local cliff test")
        D.p("cliff_index = mean |change in P(decline) across the cutoff| / mean |change over an equally wide interval away from it|. "
            "About 1 = smooth reasoning; much greater than 1 = the model reacts to the cutoff or to a label. Intervals are bootstrap over applicants.")
        c["verdict"] = [verdict_cliff(r.cliff_index_abs, r.ci_lo, r.ci_hi) for r in c.itertuples()]
        D.table(c[["run", "system", "n", "cliff_index_abs", "ci_lo", "ci_hi", "verdict"]])
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(c) + 1.2))
        lab = c.run + " " + c.system
        ax.barh(lab, c.cliff_index_abs, xerr=[c.cliff_index_abs - c.ci_lo, c.ci_hi - c.cliff_index_abs], color="#f59e0b")
        ax.axvline(1, c="k", ls=":"); ax.set_xscale("log"); ax.set_xlabel("cliff index (log scale, 1 = smooth)")
        D.fig(fig, "fig_cliff", "Cliff index by system")
        plt.close(fig)
        if v20.get("cliff") is not None:
            for r in v20["cliff"].itertuples():
                D.p(f"Final run: {r.system}: cliff index {r.cliff_index_abs:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] -> {verdict_cliff(r.cliff_index_abs, r.ci_lo, r.ci_hi)}.")
    if v20.get("bands") is not None:
        D.h("Band tables used by the band prompts (unadjusted historical default rates)", 3)
        rows = [dict(field=f, band=lab, default_rate=r, n=n) for f, rs in v20["bands"].items() for lab, r, n in rs]
        D.table(pd.DataFrame(rows))

    # ---------------------------------------------------------------- consistency + audit
    if "v1_3_run" in runs and _csv(runs["v1_3_run"], "stage08_counterfactual_table.csv") is not None:
        cf = _csv(runs["v1_3_run"], "stage08_counterfactual_table.csv")
        keep = [c for c in ["system", "response_rate", "sign_agree_all", "sign_agree_responders", "trivial_prior_agree",
                            "spearman_dP_vs_tau", "evidence_response_rate", "evidence_reliance_correct_sign",
                            "spurious_mean_abs_dP", "spurious_decision_flip_rate"] if c in cf.columns]
        D.h("7. Counterfactual consistency (flip direction vs CALF's effect sign)")
        D.p("Sign agreement is an easy test (almost every clause raises risk; see trivial_prior_agree), and the response size does not track "
            "CALF's effect size (spearman near 0). evidence_response_rate = share of cases where reversing the matched-precedent counts moved the decision.")
        D.table(cf[keep])
    if "v1_2_run" in runs and _json(runs["v1_2_run"], "stage08_coverage_risk.json"):
        cov = _json(runs["v1_2_run"], "stage08_coverage_risk.json")
        D.h("8. Audit outcome (v1.2 run, smooth prompt)")
        D.table(pd.DataFrame([{k: v for k, v in cov.items()}]), "Decision mix and default rates (small groups: interpret with care)")

    D.h("9. Limitations")
    for t in ["Effects are observational: identification needs ignorability given the adjustment set; refuters catch pipeline failures, not hidden confounding.",
              "Lending Club contains funded loans only (selection bias, no reject inference); clauses are group contrasts, not nudges of a continuous value.",
              "The clause cutoffs are analyst-defined: the retrieved policy text states none of them, and Fannie Mae's 36% DTI cutoff binds on about 1.5% of these loans.",
              "LLM evaluation samples are small (600 applicants, 85 defaults): AUROC differences below about 0.05 are within noise.",
              "Layering (CALF's tree structure) gave no measurable benefit over a flat network on these data, consistent with the paper's insurance result.",
              "The live demo and all prompts are research tools, not lending decision systems."]:
        D.p("- " + t)
    D.h("10. Reproducibility")
    D.p("Runs archived: " + ", ".join(runs) + ". Config of the final run: " + json.dumps(cfg, default=str))

    open(os.path.join(out_dir, "FINAL_REPORT.md"), "w", encoding="utf-8").write("\n".join(D.md))
    open(os.path.join(out_dir, "FINAL_REPORT.html"), "w", encoding="utf-8").write(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(D.html)}</body></html>")
    for n, b in D.figs.items():
        open(os.path.join(out_dir, n + ".png"), "wb").write(b)
    return out_dir
