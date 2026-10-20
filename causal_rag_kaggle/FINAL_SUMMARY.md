# Causal-RAG on real lending data: final summary (code v2.0)

All numbers come from the saved runs in `results/`. Data: real Lending Club loans (36-month, 2012-2015, resolved outcomes), real policy text, the real Hillstrom randomised experiment. No synthetic data.

## What works
* **CALF estimates heterogeneous clause effects well on real loans.** Calibration slope (1 = ideal) for the layered model: FICO 0.92, loan-to-income 1.18, DTI 0.86; utilization is weak (0.19). A LightGBM T-learner is badly over-dispersed (slopes 0.04-0.27).
* **Validated on a real randomised experiment (Hillstrom).** Model ATE 0.066-0.069 vs experimental 0.059 [0.045, 0.073]; CALF ranks customers far better than a T-learner (Qini about 31 vs 9).
* **Refutations pass** (placebo, random common cause, subset) for all four clauses; E-values about 1.8-2.1 (utilization fragile at 1.06).

## What did not work (reported honestly)
* **Layering gave no measurable benefit** over a flat network (same error to 3 decimals), consistent with your paper's insurance result.
* **No LLM system is clearly better than the plain LLM.** AUROC on 600 applicants (85 defaults): plain 0.633; best Causal-RAG variants 0.66-0.68 (difference +0.03 to +0.04, p about 0.06, not significant). Tabular references are at least as good: CALF factual risk 0.707, LightGBM 0.691.
* **A control that only lists which risk flags apply matches the full causal pipeline** (0.668 vs 0.678, p = 0.59): the causal evidence added nothing measurable to accuracy.
* **Band tables (unadjusted default rate by value band) did not help either**: bands only 0.647, graded + bands 0.655 (not distinguishable from 0.633).

## The cliff finding
Local cliff index (change across a cutoff / change over an equally wide interval away from it; 1 = smooth):

| system | cliff index [95% interval] | reading |
|---|---|---|
| bands only (no cutoff wording) | 0.73 [0.63, 0.84] | smooth |
| plain LLM | 1.08 [0.81, 1.49] | smooth |
| graded + bands | 2.32 [1.89, 2.90] | mild cliff |
| graded | 2.76 [2.21, 3.46] | mild cliff |
| flags only | 5.29 [4.00, 7.00] | cliff |
| smooth prompt with TRIGGERED label | 86.0 [53, 173] | strong cliff |

Every system whose prompt states conditions in terms of cutoffs or regimes reacts to the cutoff itself; the two systems without that language are smooth. Adding band tables did not remove the mild cliff of the graded prompt. The cliff is therefore tied to cutoff/regime wording (and the evidence stated in those terms), not to missing gradient information. These data cannot separate the wording from the precedent text and CALF numbers that always accompany it.

## What to conclude
Causal estimation on real data works (CALF, validated against an RCT), but feeding those estimates to an LLM did not measurably improve its lending decisions, and describing conditions as cutoffs makes its behaviour step-like. The auditing machinery (counterfactual tests, cliff test, control experiments) is the useful, reusable contribution: it detected that the "improvement" was a flag-listing effect and that some prompt designs behave like hard rules.

## Limits
Observational effects (E-values reported); funded loans only; analyst-defined cutoffs not stated in the retrieved policy text; Fannie Mae's 36% DTI binds on about 1.5% of loans; 85 defaults limit AUROC resolution (differences below about 0.05 are within noise); the live demo has not been run on Kaggle.
