# Run history (all numbers parsed from the notebooks' own saved outputs)

| run folder (results/) | code | N_EVAL | what it is | AUROC: no_rag / vanilla / kNN / causal_rag | tabular: CALF risk / LightGBM | audit: default rate approved / referred |
|---|---|---|---|---|---|---|
| `v1_0_first_full_run` | v1.0 | 300 | first full GPU run (dti_gt36, topical-only "policy support", text-pool matching) | 0.645 / 0.596 / 0.642 / 0.641 | 0.724 / 0.716 | 9.4% / 24.4% |
| `v1_1_stage01_05_cpu_session` | v1.1 | - | CPU-only session (~12 h); stages 01-05 only, kept for stage 03/04 tables | - | - | - |
| `v1_1_full_run` | v1.1 | 300 | dti_gt25, strict policy cutoff matching, balance-space matching | 0.645 / 0.596 / 0.642 / **0.707** | 0.724 / 0.716 | 9.4% / 18.5% |
| `v1_2_run` | v1.2 | 600 | smooth prompt, control experiments, resumable chunks (85 defaults) | 0.633 / 0.625 / 0.639 / 0.678 (smooth 0.666, calf-only 0.658, precedent-only 0.670) | 0.707 / 0.691 | 11.5% / 17.7% |
| `v1_3_run` | v1.3 | 600 (same bundles as v1.2) | conditions-only control, graded prompt, boundary flips for baselines | conditions-only 0.668, graded 0.660 (same no_rag/causal_rag as v1.2) | same | - (no audit stage) |
| `v1_4_cliff_run` | v1.4 | 150 | proper local cliff test | cliff index: no_rag 1.08, graded 2.76, conditions-only 5.29, smooth 86.0 | - | - |
| `v2_0_final` | v2.0 | 600 / 150 | band-table prompts + automatic final report | bands-only 0.647, graded+bands 0.655 (plain LLM 0.633); cliff index 0.73 / 2.32 | - | - |

Notes
* The sample is drawn with `random_state=0`, and with this sampler the first 300 of a 600-applicant draw are the same applicants as the 300 scored in v1.0/v1.1 (checked in my sandbox's pandas; verify via `case_id`). `FINAL_RUN_v1_2` therefore also writes `discrimination_first300.csv` for a like-for-like comparison with the older rows. Older runs stored scores only, not per-applicant data, so paired tests across runs are not possible.
* `N_EVAL=300` has only ~42 defaults: AUROC differences below ~0.07 are within noise. v1.2 uses 600.
* Each run folder holds `stage08_discrimination.csv`, `stage08_counterfactual_table.csv`, `stage08_coverage_risk.json`, logs and `RUN_INFO.md`.

Reading the v1.3 numbers (correction): the v1.3 / v1.2 'boundary flip' moves a field to just across its cutoff from wherever it is, so for applicants far from the cutoff it is a large move. Absolute 'cliff ratios' from v1.1-v1.3 are therefore confounded by move size; only comparisons across systems (same edits) are valid. The v1.4 local cliff test replaces them.

v1.4 result: the earlier 'cliff ratio ~1.0' for the smooth prompt was an artefact of the confounded metric. With the local test, the smooth prompt (explicit TRIGGERED label) is the most cliff-like system (index 86); the graded prompt cuts it to 2.8 but does not reach the plain LLM's 1.08.
