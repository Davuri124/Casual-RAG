# Full run, v1.1 code, real data (all stages 01-08 completed, no errors)
Config: SMOKE=False, N_EVAL=300 LLM-scored 2015 applicants (42 defaults, 14.0%), EconML off, Criteo skipped (HTTP 404).
Timings: stage 03 4.6 min, stage 04 20.7 min, 04b 5.5 min, stage 06 (LLM + counterfactual agent) 94 min, 07 1.3 min, 08 0.4 min.
All numbers below are parsed from the notebook's saved outputs (csv/log files in this folder).

## Discrimination (AUROC; paired bootstrap vs no_rag, n=300)
| system | AUROC | diff vs no_rag [95% CI] | p |
|---|---|---|---|
| no_rag | 0.645 | - | - |
| vanilla_rag | 0.596 | -0.049 [-0.114, +0.020] | 0.15 |
| knn_rag | 0.642 | -0.004 [-0.069, +0.063] | 0.89 |
| causal_rag | 0.707 | +0.062 [+0.001, +0.124] | 0.046 |
| CALF factual risk (tabular) | 0.724 | +0.079 [-0.013, +0.171] | 0.09 |
| LightGBM (tabular) | 0.716 | +0.071 [-0.027, +0.162] | 0.13 |
(Same 300 applicants as the v1 run: causal_rag moved 0.641 -> 0.707 with the v1.1 retrieval/policy/DTI changes; the other systems are unchanged.)

## Audit outcomes (default rates by decision; counts are small)
APPROVE 202 -> 9.4% default | REFER_TO_HUMAN 65 -> 18.5% | DECLINE 33 -> ~33% (derived: 42 total defaults minus approved and referred).
Flags: thin_overlap 83, spurious_sensitive 33, evidence_model_conflict 3, evidence_ignored 2, llm_calf_disagree 2.
Policy support: all four clauses topical-only (none state the cutoff) -> labelled analyst-defined.
Retriever: median balance SMD 0.34-0.37; thin-evidence share 26-35%.

## Counterfactual consistency (1,040 material regime-flip pairs)
| | causal_rag | no_rag | vanilla_rag |
|---|---|---|---|
| response rate | 0.446 | 0.171 | 0.140 |
| sign agreement, all | 0.809 | 0.808 | 0.734 |
| sign agreement, responders | 0.976 | 0.871 | 0.733 |
| 'every clause raises risk' baseline | 0.673 | 0.673 | 0.673 |
| Spearman(dP, tau) overall / within clause | 0.011 / 0.065 | -0.276 / 0.043 | -0.194 / -0.019 |
| mean abs dP, regime flip | 0.343 | 0.093 | 0.043 |
| mean abs dP, boundary flip (cliff) | 0.357 | - | - |
| cliff ratio boundary/regime | 1.04 | - | - |
| spurious: mean abs dP / decision-flip rate | 0.009 / 6.7% | 0.022 / 3.7% | 0.015 / 3.7% |
| evidence reversed: response rate / correct sign | 0.115 / 0.943 | - | - |

## CALF stage (2015 test, DR-calibrated; DR-MSE/BLP slope; slope 1 = well calibrated)
Layering (K=4) vs K=1 vs flat MLP: DR-MSE identical to ~3 decimals for every clause. Slopes: dti 0.86/0.53/0.94, util 0.19/0.44/0.42 (weak), fico 0.92/0.72/1.09, lti 1.18/1.47/1.23. LightGBM T-learner slopes 0.04-0.27.
Hillstrom RCT: model ATE 0.066-0.069 vs experiment 0.0588 [0.045, 0.073] (inside CI); Qini 30.7-31.7 for CALF/flat vs 9.3 LightGBM T-learner.
Refutations: placebo / random-common-cause / subset pass rates 1.0 for all clauses. E-values (point/CI): dti 2.07/1.92, util 1.32/1.06, fico 1.83/1.71, lti 1.87/1.76.
