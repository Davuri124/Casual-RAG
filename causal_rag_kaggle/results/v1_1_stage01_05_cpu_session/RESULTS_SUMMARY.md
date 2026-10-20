# Results snapshot: stages 01-05 (parsed from your notebook's own saved outputs, nothing retyped)

Run: full data (SMOKE=False), v1.1 code. Test cohort = 2015 Lending Club loans (282,887; default rate 14.9%). Train 260,136 / val 46,076.

## Stage 04 - CALF on real loans (DR-calibrated; ate_dr = doubly-robust ATE on trimmed overlap region)
| treatment   |   dr_mse |   ate_dr |   ate_dr_se |   blp_slope |   blp_slope_se |   overlap_outside |
|:------------|---------:|---------:|------------:|------------:|---------------:|------------------:|
| dti_gt25    |   1.012  |   0.0416 |      0.0021 |      0.736  |         0.108  |            0.1752 |
| util_gt75   |   1.0696 |   0.0129 |      0.0023 |      0.2528 |         0.2917 |            0.2953 |
| fico_lt680  |   0.8253 |   0.0405 |      0.0018 |      0.7429 |         0.0844 |            0.0826 |
| lti_gt20    |   0.7701 |   0.0403 |      0.0017 |      1.2566 |         0.0848 |            0.061  |

Layering benefit: DR-MSE of CALF K=4 / K=1 / flat MLP agree to the 4th decimal for every clause (full table: stage04_calf_results.csv).
LightGBM T-learner calibration slopes are 0.04-0.27 (over-dispersed); CALF/flat MLP/EconML are closer to 1.
EconML CausalForestDML DID run this time (it crashed in v1).

## Stage 04b - Hillstrom randomized experiment (held-out n = 9,600)
| model                       |   ate_model |   ate_experiment | ate_exp_ci       | ate_in_ci   |   blp_slope |     qini |
|:----------------------------|------------:|-----------------:|:-----------------|:------------|------------:|---------:|
| CALF K=4 tokens (layered)   |     0.06789 |          0.05884 | [0.0449, 0.0728] | True        |     1.0136  | 33.4879  |
| CALF K=1 (paper-as-written) |     0.06466 |          0.05884 | [0.0449, 0.0728] | True        |     1.11205 | 32.9251  |
| CALF flat (1 layer)         |     0.06753 |          0.05884 | [0.0449, 0.0728] | True        |     1.13587 | 33.3182  |
| LightGBM T-learner          |     0.05916 |          0.05884 | [0.0449, 0.0728] | True        |     0.12337 |  9.32298 |
Criteo skipped: download link returns HTTP 404.

## Stage 04b - refutations: placebo / random-common-cause / 80%-subset pass rates = 1.0 for all four clauses.
E-values (point / CI bound): dti_gt25 2.07/1.92, util_gt75 1.32/1.06, fico_lt680 1.83/1.71, lti_gt20 1.87/1.76.

## Stage 05 - retriever (evidence bundles for the sampled test applicants; evidence.pkl itself was LOST with the session)
Median balance SMD: dti 0.33, util 0.41, fico 0.35, lti 0.35 (v1: 2.63 / 0.71 / 0.68 / 0.49).
Thin-evidence share: dti 25%, util 34%, fico 27%, lti 27% (v1: 98% / 65% / 65% / 48%).
Policy: all four clauses are topical-only (no retrieved chunk states the cutoff) -> labelled analyst-defined.

## What this run does NOT contain
Stages 06-08 (LLM scoring, counterfactual agent, audit, final evaluation) have not been run on v1.1.
Binary artifacts (evidence.pkl, calf_scores.parquet, calf_models.pt, featurizers.pkl, embeddings) cannot be rebuilt from printed output.

## Timing warning
Stage 03 took 367 min and stage 04 took 146 min; in your earlier run they took ~5 and ~21 min. That pattern means this session had no working GPU.
