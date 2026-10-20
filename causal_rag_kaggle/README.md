# START HERE - Causal-RAG final version (v2.0)

**Two notebooks are all you need** (in `notebooks/`):

| Notebook | When | Time |
|---|---|---|
| `FINAL_v2_0_last_run.ipynb` | the last experiment + automatic final report | about 1.5-2 h on a T4 (estimate; resumable) |
| `LIVE_DEMO.ipynb` | optional interactive demo with a public link | about 5 min to load |

Everything else (`notebooks/archive_earlier_versions/`) is kept only for reproducing earlier runs.

## Steps for the final run
1. **Code dataset**: upload `causal_rag_kaggle.zip` as a new dataset version (it already contains all earlier results in `results/`). Attach only one copy.
2. **Artifacts**: you already have them in your earlier notebook's Output (`evidence.pkl`, `lc_clean.parquet`, ...). In the new notebook choose **Add Input -> Notebooks / Your Work -> that notebook** (no download needed).
3. Import `FINAL_v2_0_last_run.ipynb`; you do NOT need the Lending Club dataset (the cleaned data is in the artifacts). Internet ON, GPU T4 x2.
4. Run it as **Save Version -> Save & Run All (Commit)**. Leave `SMOKE = False`. Close the laptop.
5. When finished, open the version's **Output**: download `FINAL_RESULTS_v2_0.zip` (contains `FINAL_REPORT.html`, `FINAL_REPORT.md`, figures and tables). That is the finished project write-up.

What the last run adds: (1) `bands_only` (applicant + unadjusted default-rate-by-band tables, no cutoff/flag/regime wording) and (2) `causal_graded_bands` (graded Causal-RAG prompt + the same tables), each scored for AUROC on the same 600 applicants and put through the local cliff test; then it builds the final report from ALL archived runs plus this one.

If a run stops, just run the notebook again: it resumes from the last saved chunk.

---

# Causal-RAG v1.4: counterfactual-audited retrieval for high-stakes decisions (Lending Club + CALF)

Real data only: Lending Club accepted loans (Kaggle), Fannie Mae Selling Guide + CFPB Reg Z / Reg B (official sites), Hillstrom RCT.

| Agent | Module |
|---|---|
| Retriever (dense + structural causal graph, balance-space matched precedent) | `retrieval.py`, `causal_graph.py`, `policy_corpus.py` |
| Causal Attribution (your CALF) | `calf.py`, `pipeline.run_calf_stage` |
| Counterfactual Perturbation | `counterfactual.py` |
| Synthesis & Audit (decision + causal DAG) | `audit.py` |

## Use on Kaggle (3 steps)
1. **Dataset**: upload `causal_rag_kaggle.zip` as a new dataset (or a New Version of your existing one). It contains the code AND `results/` (every earlier run). Make sure only ONE copy of the dataset is attached.
2. **Notebook**: import `notebooks/FINAL_RUN_v1_2.ipynb`. Add inputs: the dataset + `wordsforthewise/lending-club`. Internet ON, GPU T4 x2.
3. **Run**: first with `SMOKE = True` (dry run, minutes), then `SMOKE = False` and **Save Version -> Save & Run All (Commit)**. The setup cell prints `GPU: ...` and refuses to start a multi-hour run on CPU.

Estimated time at `N_EVAL = 600` with a working GPU: stages 01-05 about 35 min, LLM stage about 4.5-5 h (my estimate from the v1.1 run: 94 min for 300 applicants with 3+4 systems; v1.2 scores 7 systems and runs counterfactual tests on 4). It saves every `CHUNK` (100) applicants and resumes if re-run.

## Keeping results and adding new ones
* Stage 08 also writes `discrimination_first300.csv`: the first 300 applicants are the same ones scored in v1.0/v1.1, so compare that table with the older runs.
* `results/<run_name>/` holds each finished run (tables, logs, `RUN_INFO.md`). `FINAL_RUN_v1_2` automatically reads them and prints a table across all runs (`all_runs_*.csv`).
* After a run, open the version's Output tab -> `artifacts/results_export/<RUN_NAME>/`. Copy that folder into the dataset's `results/` and publish a new dataset version. Next time the comparison includes it.
* Change `RUN_NAME` in the setup cell for each new run.
* If your commit's Output still has `calf_scores.parquet`, `calf_models.pt`, `featurizers.pkl`, `case_emb.npy`, `policy_emb.npy`, `policy_chunks.parquet`, `case_cards.parquet`, `lc_clean.parquet`, `graph.json`, `adjustment_sets.json`: add it as an input and stages 01-04 skip themselves.

## What v1.2 adds (and what it tests)
* **Control experiments**: `causal_calf_only` (CALF numbers, no matched precedent) and `causal_precedent_only` (precedent, no CALF numbers). They show what drives Causal-RAG's AUROC (see `pairwise_table.csv`).
* **Smooth prompt** (`causal_rag_smooth`): all four conditions always shown with the applicant's value and a TRIGGERED / not-triggered status, plus a sentence that cutoffs are conventions, not cliffs. Tests whether the cliff (boundary-flip response ~ regime-flip response) goes away. It changes layout AND adds that sentence, so a positive result does not isolate which one helped.
* **Policy fix**: the "utilization" entity no longer matches generic words like "revolving"/"credit card".
* **Resumable LLM stage**, N_EVAL 600, run-history comparison and export.

## Known limits (unchanged)
Observational effects (E-values reported, util clause fragile); funded loans only; clauses are group contrasts; layering has not beaten a flat network; LightGBM/CALF tabular risk is a strong reference that the LLM systems have not clearly beaten; the Causal-RAG prompt carries information from a model trained on defaults (the control experiments address this); Criteo link is dead (HTTP 404).

## Verification status
Executed on Kaggle (user): v1.0 and v1.1 end to end. v1.2 changes have been tested only with a stand-in LLM/embedder (prompts, chunked resume, comparison, export, all notebooks parse). The Gradio demo cell has never been run.


## v1.3: next-step experiments + live demo

New notebooks (both need the **artifacts** of a finished run attached as a dataset, so keep your run's `artifacts/` folder:
Output tab -> New Dataset, or run the "package artifacts" cell at the end of `FINAL_RUN_v1_2` / `NEXT_v1_3`):

| Notebook | What it does | Time (estimate) |
|---|---|---|
| `NEXT_v1_3_new_controls.ipynb` | (1) **conditions-only control**, (2) **graded prompt** (no TRIGGERED word), (3) **boundary flips for no-RAG / vanilla** on the SAME 600 evidence bundles; compares with the v1.2 scores, exports `v1_3_run` | about 3-4 h at 600 applicants |
| `LIVE_DEMO.ipynb` | Gradio app with a public link: pick a real 2015 loan, edit DTI / utilization / FICO / amounts, run the audit (decision, flags, causal DAG, per-condition table) and a **what-if sweep** plot of P(decline) vs a field for four prompt designs | loading about 5 min, each audit about 15-30 s |

How to read the new experiment:
* `causal_conditions_only` vs `no_rag` significant, and `causal_rag` not better than it -> the AUROC gain is just "the flags were listed".
* `cliff_ratio_boundary_over_regime` near 1 -> the system reacts to the label/regime, near 0 -> smooth. If `causal_graded` is clearly below `causal_rag_smooth`, the TRIGGERED word was driving the cliff. Baseline cliff ratios use the regime means from the v1.2 run.
* The demo's sweep tab shows the same thing visually (step = cliff, ramp = smooth).

Demo notes: `launch(share=True, debug=True)` prints a temporary public *.gradio.live link and keeps the cell running; stop the cell to shut it down, and expect the link to die when the Kaggle session ends. Needs GPU + Internet. Research demo only, not a lending decision tool. The what-if sweep holds the retrieved evidence fixed (it re-prompts, it does not re-retrieve).

Verification: new prompts, boundary-only mode, the live auditor (analysis, edits, all four sweeps), the Gradio app construction and its callbacks, and the NEXT notebook cells were executed in a sandbox with a stand-in LLM. They have NOT run on Kaggle with the real model, and the public share link has never been launched.


## v1.4: corrected cliff test
`CLIFF_TEST_v1_4.ipynb` replaces the boundary-flip "cliff ratio". For each clause it asks the LLM at thr-d, thr+d (just across the cutoff) and at thr-5d, thr-3d (an equally wide interval away from it, the placebo), all other fields unchanged.
`cliff_index = mean|change across cutoff| / mean|change over placebo interval|` (1 = smooth, much greater than 1 = reacts to the cutoff or a label), with bootstrap intervals over applicants. Needs `evidence.pkl` from your run attached. Estimated 2 h for 150 applicants x 4 systems. Tested only with stand-in LLMs (a smooth one gives about 1.0-1.1, a deliberately label-driven one gives 10-30).
