# Causal-RAG

**Counterfactual-audited retrieval for high-stakes lending decisions.**

Causal-RAG is a four-agent system that retrieves *causally adjusted* evidence for an underwriting decision, stress-tests a language model's answer with counterfactual edits, and emits an auditable decision record (APPROVE / DECLINE / REFER_TO_HUMAN) with a causal graph instead of a wall of text. Causal effects are estimated with **CALF**, a tree-structured causal data fabric for individual treatment effect (ITE) estimation.

> **Real data only.** Lending Club accepted loans (2012-2015), Fannie Mae and CFPB policy documents, and the Hillstrom randomised e-mail experiment. No synthetic data anywhere in the pipeline.

**Status:** version 2.0.0 (final) | seven archived runs (v1.0, v1.1 x2, v1.2, v1.3, v1.4, v2.0) | compute: Kaggle notebooks (GPU T4) | LLM: Qwen2.5-7B-Instruct (4-bit NF4)

---

## Table of contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Research questions and headline findings](#research-questions-and-headline-findings)
4. [Data](#data)
5. [Method](#method)
6. [Results](#results)
7. [Problems met and how they were solved](#problems-met-and-how-they-were-solved)
8. [Run history](#run-history)
9. [Repository layout](#repository-layout)
10. [Installation and reproduction](#installation-and-reproduction)
11. [Live demo](#live-demo)
12. [Limitations](#limitations)
13. [Future work](#future-work)
14. [Author and citation](#author-and-citation)

---

## Motivation

Standard retrieval-augmented generation (RAG) finds documents that *look* similar, but similarity is not causation. In automated underwriting, medical protocols or policy compliance, matching similar historical cases tends to repeat past biases and to mistake confounders for true risk drivers. Banks and healthcare enterprises also face a compliance and interpretability problem when deploying LLMs.

Causal-RAG addresses this by making the evidence causally adjusted, the LLM's behaviour counterfactually tested, and the final output auditable.

---

## Architecture

| Agent | Role |
|---|---|
| **Retriever Agent** | Dense retrieval plus a structural causal graph and balance-space matched precedent (like-for-like historical loans). Also retrieves policy text linked to each clause through entity tags. |
| **Causal Attribution Agent** | CALF estimates the individual effect of each underwriting clause on default. |
| **Counterfactual Perturbation Agent** | Edits the applicant or the evidence and checks whether the LLM reacts the way the causal estimate says it should. |
| **Synthesis & Audit Agent** | Deterministic decision rule plus audit flags, a JSON record, a causal DAG image, an HTML report and an LLM-written rationale (kept separate from the audit table). |

Around the agents sit the validation tools: doubly-robust calibration tests, a real randomised experiment, refutation tests, E-values, control experiments and a local "cliff" test.

---

## Research questions and headline findings

| RQ | Question | Answer |
|---|---|---|
| **RQ1** | Can CALF credibly estimate the effect of individual underwriting clauses on default using real data? | **Yes, with caveats.** Well calibrated for FICO, loan-to-income and DTI; weak for utilization. Validated against a real RCT. |
| **RQ2** | Does causally adjusted evidence improve LLM lending decisions vs. plain prompting, vanilla RAG and kNN RAG? | **Not measurably.** |
| **RQ3** | Is the LLM consistent with the causal estimates under counterfactual perturbation? | **Only in direction.** Effect sizes are not tracked; cutoff wording induces step-like behaviour. |
| **RQ4** | Can this be turned into an auditable decision record a human can contest? | **Yes, with modest benefit.** Decisions are monotone in realised risk. |

### Key findings

- **CALF works on real loans.** Calibration slope (1 = ideal): FICO 0.92, loan-to-income 1.18, DTI 0.86, utilization 0.19 (weak). A LightGBM T-learner is badly over-dispersed (slopes 0.04-0.27).
- **Validated on a randomised experiment (Hillstrom).** Model ATE 0.066-0.069 lies inside the experimental interval 0.059 [0.045, 0.073]. CALF ranks customers far better than a T-learner (Qini about 31 vs 9).
- **All refutation tests passed** (placebo treatment, random common cause, subsample) for all four clauses. E-values are 1.8-2.1 for DTI, FICO and loan-to-income; utilization is fragile (1.06).
- **No LLM system was clearly better than the plain LLM.** On 600 held-out 2015 applicants (85 defaults), AUROC was 0.633 for the plain LLM and 0.658-0.678 for Causal-RAG variants (best p = 0.06). Tabular models were at least as good (CALF risk 0.707, LightGBM 0.691).
- **A flags-only control matched the full pipeline** (0.668 vs 0.678, p = 0.59): the causal evidence added no measurable accuracy beyond listing which standard risk flags apply.
- **Cutoff wording makes the LLM step-like.** Plain reasoning is smooth (cliff index 1.08), as is a bands-only prompt (0.73). The flag-label prompt jumps 86x more across the cutoff than away from it. The graded prompt reduces the cliff to 2.8 (2.3 with band tables) but does not remove it.
- **The audit machinery is the reusable contribution.** It exposed a flag-listing effect, showed that sign agreement is an easy test, and caught mistakes in our own metrics.

**Bottom line:** causal effect estimation on real data works and can be validated honestly. Feeding those estimates to an LLM did not measurably improve its lending decisions, and describing conditions as cutoffs makes its behaviour step-like. Negative results and corrected mistakes are reported as measured.

---

## Data

### Lending Club accepted loans (main dataset)

Kaggle `wordsforthewise/lending-club` (accepted loans, 2007-2018). Deliberate preparation choices (`data_prep.py`):

- **Pre-origination fields only** (known at application time). No payment-history or recovery columns (post-outcome leakage).
- **Lender risk outputs excluded:** `grade`, `sub_grade`, `int_rate` encode the previous underwriting decision.
- **36-month loans issued 2012-2015** with a resolved outcome (Fully Paid = 0, Charged Off = 1). Loans flagged "does not meet the credit policy" are excluded.
- **Right-censoring guard:** the unresolved share never exceeded 0.32% (December 2015); the pipeline would trim the test cohort above 1%.
- **Temporal split:** train = 85% of 2012-2014, validation = random 15% of 2012-2014, test = all of 2015.

| Role | Loans | Default rate |
|---|---|---|
| Train (2012-2014) | 260,136 | 0.1325 |
| Validation (random 15% of 2012-2014) | 46,076 | 0.1320 |
| Test (2015) | 282,887 | 0.1488 |

A **306,212-loan retrieval index** is built from train + validation loans (each loan rendered as a text "card" and embedded).

### Policy documents

Four public documents chunked into 88 passages: Fannie Mae Selling Guide B3-6-02 (Debt-to-Income Ratios) and B3-5.1-01 (Credit Scores), CFPB / eCFR 12 CFR 1026.43 (ability to repay) and 12 CFR 1002.6 (Regulation B). Passages are tagged with entities (income, repayment, DTI, credit score, utilization, prohibited basis). Add your own documents through a `policy_docs/` folder.

### Hillstrom e-mail experiment

Randomised experiment used for validation: "any e-mail vs none" as treatment (known propensity about 2/3), website visit as outcome. The Criteo uplift dataset was planned as a second experiment but its download link now returns HTTP 404, so it is skipped.

---

## Method

### Mapping CALF's layers to a loan application

CALF splits covariates into Roots, Soil, Leaves and Choice. These are mapped to temporal/causal tiers of a loan application. A treatment lives in one tier; its adjustment set is every covariate in strictly earlier tiers plus same-tier covariates not part of, or mechanically tied to, the treatment.

| Tier | Columns |
|---|---|
| 0 Roots | log annual income, employment years (+ missing flag), credit-history length, home ownership |
| 1 Soil | state, income-verification status (flagged as a judgment call) |
| 2 Leaves | FICO, delinquencies (2y), inquiries (6m), open accounts, public records, total accounts, mortgage accounts, revolving balance (log), revolving utilization, DTI |
| 3 Choice | log loan amount, loan-to-income, loan purpose |
| 4 Outcome | default (charged off) |

### The four underwriting clauses (treatments)

Each clause is a binary "regime"; the estimated effect is the individual effect on default probability of being in the regime rather than outside it, for the same background profile. Columns defining the treatment are removed from inputs to preserve overlap.

| Key | Clause | Treated share | Cutoff origin |
|---|---|---|---|
| `dti_gt25` | DTI above 25% | 0.205 | analyst-defined |
| `util_gt75` | Revolving utilization above 75% | 0.214 | analyst-defined |
| `fico_lt680` | FICO below 680 | 0.361 | analyst-defined (Fannie Mae minimum is 620) |
| `lti_gt20` | Loan amount above 20% of income | 0.427 | analyst-defined |

> **Design mistake found and fixed.** The first version used Fannie Mae's 36% DTI guideline. Only 1.5% of Lending Club loans exceed it, so 85% of test loans had propensity below 0.05 (positivity failure). The clause became `dti_gt25`. That the 36% cutoff binds on almost nobody is itself a finding.

### Structural causal graph

Built with a "discovery plus knowledge" recipe: (1) PC algorithm (causal-learn, Fisher-z) on a 30,000-loan sample; (2) temporal-tier background knowledge (no edge from a later tier to an earlier one); (3) expert and mechanical edges for definitional relations (loan-to-income = amount / income, etc.) and for clause-to-default edges. Two adjustment sets are produced per clause: `A_default` (tier rule) and `A_graph` (additionally drops descendants and direct neighbours of the treatment: possible mediators, colliders, "bad controls"). CALF is fitted on both and the difference is reported as a sensitivity analysis.

### Adapting CALF to real data

| Aspect | Change and reason |
|---|---|
| Outcome | Binary default: cross-entropy on logits; effect = p1 - p0 (risk difference). |
| Model selection | Checkpoints selected on MSE against a cross-fitted doubly-robust pseudo-outcome (PEHE needs unobservable true effects). |
| Monotonicity prior | Set to 0 by default; baking in "effects are non-negative" would make the audit partly circular. |
| Layer count | Variable (tiers 0 to the treatment tier). |
| Attention | The paper's single-vector layers make the softmax identically 1 (a linear map). Added `k_tokens` (4 latent tokens per layer); `k_tokens = 1` reproduces the paper as an ablation. |
| Training | Batch 2,048, up to 40 epochs, early stopping. Ensemble of 3 seeds; between-seed spread replaces MC-Dropout (which under-covers). |
| What the LLM may see | CALF effect estimates only, never CALF's factual risk prediction, so AUROC gains cannot be risk leakage. |

### Evaluating effects without ground truth

1. **Doubly-robust (AIPW) pseudo-outcome**, cross-fitted with LightGBM nuisances: DR-MSE ranks models like PEHE would.
2. **BLP calibration slope** (1 = real, correctly scaled heterogeneity; 0 = no signal) and GATES quintile tables.
3. **Real randomised experiment** (Hillstrom): known propensity, experimental difference in means, Qini coefficient.
4. **Refutations and sensitivity:** placebo treatment, random common cause, 80% subsets, E-values.

### Retriever Agent

1. **Dense stage:** text cards embedded with `BAAI/bge-small-en-v1.5`; top 2,000 labelled pre-2015 loans form the candidate pool.
2. **Causal stage:** like-for-like matching in a 2-D balance space (logit propensity and logit prognostic risk from CALF's heads, on the graph-defined adjustment set). The 25 nearest treated and 25 nearest control loans are selected. Balance-space neighbours across the whole index are added to the pool (v1.1).
3. **Local effect:** matched default-rate difference with CI, SMD of the balance scores, and a "thin evidence" flag (fewer than 12 loans in an arm, or SMD > 0.5).
4. **Policy stage:** dense retrieval restricted to passages whose entity tags link to the clause variable. A clause is **policy-supported** only if the retrieved text states its cutoff (regex); otherwise it is labelled "topical only / analyst-defined".

### LLM scoring and prompt systems

Decisions are not parsed from free text. Each prompt ends with *"A = approve, B = decline, answer with one letter"* and the score is the next-token logit margin `logit(B) - logit(A)`: deterministic, continuous, usable for AUROC, calibration and counterfactual deltas, immune to formatting failures. Free text is generated only for the audit rationale.

Eleven prompt systems were compared on the same applicants:

| System | What the model sees |
|---|---|
| `no_rag` | Applicant card only (plain LLM reference). |
| `vanilla_rag` | Card + top policy passages + 6 most text-similar past loans with outcomes. |
| `knn_rag` | Card + 6 nearest past loans in feature space, with outcomes. |
| `causal_rag` | For each clause that applies: policy basis, matched precedent, CALF effect with spread. |
| `causal_rag_smooth` | All four clauses always shown, with value and TRIGGERED / not triggered label, plus a "cutoffs are conventions, not cliffs" sentence. |
| `causal_calf_only` | Control: CALF numbers, no precedent. |
| `causal_precedent_only` | Control: matched precedent, no CALF numbers. |
| `causal_conditions_only` | Control: list of applicable standard risk flags only. |
| `causal_graded` | Like smooth, without the TRIGGERED word: value and signed distance to cutoff. |
| `bands_only` | Control: applicant + unadjusted historical default rate by value band; no cutoffs, flags, regimes, precedent or CALF. |
| `causal_graded_bands` | Graded prompt plus band tables. |

### Counterfactual Perturbation Agent

Re-asks the same LLM with the same template after editing the applicant or the evidence, and compares the change in P(decline) with CALF's effect:

- **Regime flip** `do(T -> 1 - T)`: the clause field moves to the median of the opposite matched arm. The expected sign is direction times the sign of CALF's effect.
- **Boundary flip:** field moved just across the cutoff (later found to be confounded by move size; see the cliff test).
- **Evidence reversal / removal:** swap the matched-precedent counts between arms; a model that uses the evidence must move opposite to the original local effect.
- **Spurious edits:** state, employer title and loan title replaced by real random values; the decision should not move.
- **Local cliff test** (v1.4): see below.

### Local cliff test

For each clause and applicant, the LLM is queried just below and just above the cutoff (`thr - d`, `thr + d`) and over an equally wide placebo interval far from it (`thr - 5d`, `thr - 3d`).

`cliff index = mean |change in P(decline) across the cutoff| / mean |change over the placebo interval|`

An index near 1 means smooth reasoning; much greater than 1 means the model reacts to a label or regime. Intervals are bootstrapped over applicants.

### Synthesis & Audit Agent

The decision rule is deterministic given its inputs; the LLM supplies only the margin and a rationale. Base decision: DECLINE if the margin exceeds a label-free threshold (riskiest 20% of the sample), else APPROVE. Five counted audit flags:

| Flag | Meaning |
|---|---|
| `llm_calf_disagree` | LLM moved opposite to CALF's sign on a regime flip |
| `evidence_model_conflict` | Matched precedent and CALF disagree in sign (CI excluding 0) |
| `thin_overlap` | Thin or unbalanced evidence, or propensity outside 0.05-0.95 |
| `evidence_ignored` | LLM moved the wrong way when the evidence was reversed |
| `spurious_sensitive` | Decision changes when only irrelevant fields change |

`policy_unsupported` is informational only. The decision becomes **REFER_TO_HUMAN** if at least two flags are raised, or one flag with a margin near the threshold. Each case produces a JSON record (decision, flags, per-clause table, evidence case ids, prohibited-basis input check), a causal DAG image with CALF effects on clause-to-default edges coloured by whether the LLM agreed, a self-contained HTML report, and an LLM-written rationale clearly separated from the audit table.

---

## Results

### RQ1: CALF on real loans (2015 test cohort)

| Clause | CALF layered (K=4) BLP slope | CALF K=1 | Flat MLP | LightGBM T-learner | ATE (DR) |
|---|---|---|---|---|---|
| `fico_lt680` | 0.92 | 0.72 | 1.09 | 0.16 | 0.0405 |
| `lti_gt20` | 1.18 | 1.47 | 1.23 | 0.27 | 0.0403 |
| `dti_gt25` | 0.86 | 0.53 | 0.94 | 0.19 | 0.0416 |
| `util_gt75` | 0.19 | 0.44 | 0.42 | 0.04 | 0.0129 |

- **Layering did not help.** CALF (K=4), CALF (K=1) and a flat MLP have the same DR-MSE to three decimals for every clause, repeating the paper's own insurance finding.
- **The T-learner is over-dispersed**, producing far more heterogeneity than the data support.
- **Adjustment-set sensitivity.** The graph-derived set leaves average effects nearly unchanged but lowers calibration slopes for DTI (0.54) and FICO (0.56), so both sets are reported.

### Validation on a real randomised experiment (Hillstrom)

| Model | Model ATE | Experiment ATE (95% CI) | BLP slope | Qini |
|---|---|---|---|---|
| CALF K=4 (layered) | 0.0686 | 0.0588 [0.0449, 0.0728] | 0.90 | 30.7 |
| CALF K=1 (paper as written) | 0.0674 | same | 1.06 | 31.3 |
| CALF flat (1 layer) | 0.0657 | same | 1.09 | 31.7 |
| LightGBM T-learner | 0.0592 | same | 0.12 | 9.3 |

All models recover the average effect, but only the neural models rank customers by benefit.

### Refutations and E-values (full 2015 test set)

| Clause | DR effect | SE | E-value (point) | E-value (CI bound) |
|---|---|---|---|---|
| `dti_gt25` | 0.0511 | 0.0047 | 2.07 | 1.92 |
| `fico_lt680` | 0.0364 | 0.0037 | 1.83 | 1.71 |
| `lti_gt20` | 0.0385 | 0.0035 | 1.87 | 1.76 |
| `util_gt75` | 0.0090 | 0.0043 | 1.32 | 1.06 |

Passing the refuters rules out gross pipeline failure; it does not prove ignorability.

### Retriever evidence quality

Adding balance-space neighbours (v1.1) cut the thin-evidence share from 48-98% to 25-34% and the median SMD from 0.49-2.63 to about 0.33-0.41.

### RQ2: do LLM systems benefit from causal evidence? (600 applicants, 85 defaults)

| System | AUROC | Diff vs plain LLM | p |
|---|---|---|---|
| `no_rag` (plain LLM) | 0.633 | - | - |
| `vanilla_rag` | 0.626 | -0.008 | 0.768 |
| `knn_rag` | 0.639 | +0.005 | 0.774 |
| `causal_rag` | 0.678 | +0.044 | 0.060 |
| `causal_rag_smooth` | 0.666 | +0.033 | 0.164 |
| `causal_calf_only` | 0.658 | +0.024 | 0.240 |
| `causal_precedent_only` | 0.670 | +0.036 | 0.062 |
| `causal_conditions_only` | 0.668 | +0.035 | 0.068 |
| `causal_graded` | 0.660 | +0.027 | 0.258 |
| `bands_only` | 0.647 | +0.013 | 0.222 |
| `causal_graded_bands` | 0.655 | +0.022 | 0.360 |
| CALF factual risk (tabular) | 0.707 | +0.073 | 0.028 |
| LightGBM (tabular ceiling) | 0.691 | +0.057 | 0.106 |

- **Winner's curse.** On the first 300 applicants (42 defaults) Causal-RAG scored 0.707-0.710; on the full 600 it is 0.678.
- **Flags-only control** matches the full pipeline (0.668 vs 0.678, p = 0.59). The most likely explanation is a feature-engineering effect: the flags are real risk indicators the plain LLM does not compute on its own. This is a hypothesis; the experiments rule out the causal evidence as the driver but do not prove the flag explanation.
- **The project's claim is therefore about causal consistency and auditability, not about beating tabular models.**

### RQ3: counterfactual consistency (2,090 material regime-flip pairs)

- **Sign agreement is an easy test.** A rule that always assumes "a clause raises risk" scores 0.68; the plain LLM already scores 0.81.
- **Response size does not track effect size** (Spearman near zero or negative for every system).
- **The LLM largely ignores matched-precedent counts** in the v1.1 and smooth prompts (moves in only 12-14% of reversals); the graded prompt raises this to 37%, with 96% correct direction.
- **Irrelevant edits flip 3-5% of decisions** for every system.

### RQ4: the audit agent (v1.2, n = 600)

| Decision group | Realised default rate |
|---|---|
| Approved | 11.5% |
| Referred to human | 17.7% |
| Declined | about 23% |

Ordering is monotone and sensible. Approving the referred cases too would raise the approved-group default rate from 11.5% to 12.5%, so referral removes about one percentage point of risk. Flag counts: `thin_overlap` 162, `spurious_sensitive` 39, `evidence_model_conflict` 8, `evidence_ignored` 5, `llm_calf_disagree` 4. Most referrals reflect weak evidence rather than detected inconsistency.

### The cliff test (150 applicants x 4 clauses)

| System | Cliff index | 95% CI | Reading |
|---|---|---|---|
| `bands_only` | 0.73 | [0.63, 0.84] | smooth |
| `no_rag` | 1.09 | [0.81, 1.49] | smooth |
| `causal_graded_bands` | 2.32 | [1.89, 2.90] | mild cliff |
| `causal_graded` | 2.76 | [2.21, 3.46] | mild cliff |
| `causal_conditions_only` | 5.29 | [4.00, 7.00] | cliff |
| `causal_rag_smooth` | 86.0 | [53.3, 172.9] | cliff |

- The smooth prompt, *designed* to remove the cliff, is the most cliff-like system: its answer is nearly flat until the TRIGGERED label flips, then jumps by about 0.33 in P(decline).
- Every system whose prompt describes conditions in terms of cutoffs or regimes reacts to the cutoff itself; the two systems without such language are smooth.
- This matters because real risk changes continuously across 25% DTI (adjacent bands differ by only about 1-5 percentage points of default rate), yet label-driven prompts treat 24.9% and 25.1% very differently, exactly the hard-rule behaviour an auditor wants to detect.
- A mistake in our own earlier "boundary flip" metric (confounded by move size) was caught and retracted; the local cliff test replaced it.

---

## Problems met and how they were solved

Real data and real infrastructure produced many problems, listed in full because several changed the conclusions.

**Methodological**

| # | Problem | Solution / outcome |
|---|---|---|
| 1 | No ground-truth individual effects, so PEHE cannot be computed. | Doubly-robust pseudo-outcomes (DR-MSE, BLP, quintiles), a real RCT, refuters, E-values. |
| 2 | The paper selects checkpoints by validation PEHE. | Select by MSE against a cross-fitted DR pseudo-outcome. |
| 3 | Leakage from grade, sub-grade, interest rate and payment/recovery columns. | Only pre-origination columns kept. |
| 4 | Defining column left in inputs collapses overlap. | Per-clause drop lists; stricter graph-derived adjustment set for sensitivity. |
| 5 | 36% DTI clause binds on 1.5% of loans; CALF slope -0.16. | Redefined as `dti_gt25`; rarity reported as a finding. |
| 6 | Paper's cross-attention has a single key (softmax identically 1). | Added `k_tokens`; K=1 reproduces the paper; no measurable difference here. |
| 7 | A "non-negative effects" prior makes the audit circular. | Monotonicity weight set to 0. |
| 8 | MC-Dropout intervals under-cover (66.9% vs 90%). | Seed ensembles with between-seed spread. |
| 9 | "Policy-supported" only meant topically related. | Support requires the retrieved text to state the cutoff (regex). |
| 10 | Matched evidence thin or unbalanced. | Balance-space neighbours across the whole index. |
| 11 | 300 applicants looked significant (0.707, p = 0.046); 600 did not (0.678). | Doubled the sample, reported intervals, kept a like-for-like table, named it a winner's curse. |
| 12 | Sign agreement with CALF looked excellent but is easy. | Added the "every clause raises risk" baseline and Spearman effect-size correlation. |
| 13 | Possible CALF leakage into the LLM prompt. | LLM sees effect estimates, never CALF's factual risk; tabular rows kept separate. |
| 14 | Was the gain from causal evidence or just the flags? | Control systems: flags-only matched the full pipeline. |
| 15 | Boundary-flip cliff metric confounded by move size. | Replaced by the local cliff test; earlier reading retracted. |
| 16 | Graded prompt and bands only partly removed the cliff. | Reported as a residual cliff tied to cutoff/regime wording. |

**Engineering and environment**

| # | Problem | Solution / outcome |
|---|---|---|
| E1 | EconML causal-forest baseline crashed. | Fixed; off by default (very slow). |
| E2 | Criteo download link returns HTTP 404. | Skipped gracefully; Hillstrom is the validation experiment. |
| E3 | Hillstrom loader accepted a wrong file. | Validates required columns, falls back to the original site; RCT step is non-fatal. |
| E4 | A session without a working GPU took about 12 hours for 5 stages. | Setup cell checks the GPU and refuses long stages on CPU; stages skip themselves when outputs exist. |
| E5 | Notebook imported an old or incomplete copy of the code. | Setup cell lists copies, picks a complete and current one, clears stale imports. |
| E6 | Results lost when sessions ended; restore copied 0 files. | Recursive restore, attachable notebook outputs, packaging cell, archived run history. |
| E7 | Long LLM stages risked timeouts. | Chunked, resumable stages (100 applicants per chunk; 25 for the cliff test). |
| E8 | GPU daily quota and 12-hour session limits. | Commit versions sized to fit (LLM stages 83-351 min; final run 143 min). |

---

## Run history

| Version | What changed | Applicants / defaults | Key result | LLM stage (T4) |
|---|---|---|---|---|
| v1.0 | First full GPU run; `dti_gt36`, topical-only policy support, text-pool matching | 300 / 42 | Causal-RAG 0.641 (no gain); evidence thin in 48-98% | 83 min |
| v1.1 (CPU session) | Stages 1-5 only, no working GPU | - | Retriever fixes confirmed | about 12 h |
| v1.1 | `dti_gt25`, strict cutoff matching, balance-space matching | 300 / 42 | Causal-RAG 0.707 (+0.062, p = 0.046) | 94 min |
| v1.2 | Smooth prompt, CALF-only and precedent-only controls, resumable chunks | 600 / 85 | Causal-RAG 0.678 (+0.044, p = 0.06) | 351 min |
| v1.3 | Flags-only control, graded prompt | 600 / 85 | Flags-only 0.668 matches Causal-RAG | 295 min |
| v1.4 | Local cliff test | 150 x 4 clauses | Cliff index: plain 1.08, graded 2.76, flags 5.29, smooth 86 | 125 min |
| v2.0 | Band-table prompts and automatic final report | 600 / 85 and 150 | Bands-only 0.647; graded + bands 0.655, cliff 2.32 | 143 min |

---

## Repository layout

```
causal_rag/                    # Python package (about 20 modules)
  data_prep.py                 # Lending Club cleaning, splits, censoring guard
  config.py                    # tiers, clauses, featurizer
  causal_graph.py              # PC + tier rules + expert edges, adjustment sets
  policy_corpus.py             # policy download, chunking, entity tagging
  embedding.py  retrieval.py   # dense stage, balance-space matching
  calf.py                      # CALF (K tokens, ensembles, DR checkpointing)
  dr_eval.py                   # doubly-robust pseudo-outcome, BLP, GATES
  refute.py  rct_validation.py # refuters, E-values, Hillstrom validation
  prompts.py  llm.py           # 11 prompt systems, logit-margin scoring
  counterfactual.py            # regime flips, evidence reversal, spurious edits, cliff test
  audit.py                     # flags, decision rule, JSON/DAG/HTML records
  eval.py  pipeline.py         # AUROC, paired bootstrap, orchestration
  compare.py  report.py        # run-history comparison, automatic report
  live_demo.py  paths.py       # Gradio app, path handling
notebooks/
  01-08 stage notebooks, v1.2 / v1.3 / v1.4 experiment notebooks
  FINAL_v2_0_last_run.ipynb    # last experiment + automatic report
  LIVE_DEMO.ipynb              # Gradio demo
  archive_earlier_versions/    # reproduce earlier runs
results/                       # archived runs (tables, logs, figures)
  v1_0_first_full_run/  v1_1_stage01_05_cpu_session/  v1_1_full_run/
  v1_2_run/  v1_3_run/  v1_4_cliff_run/  v2_0_final/
policy_docs/                   # optional: add your own policy documents
README.md  RESULTS_INDEX  FINAL_SUMMARY
```

---

## Installation and reproduction

**Stack:** PyTorch, LightGBM, causal-learn, sentence-transformers, bitsandbytes, Gradio, pandas, scikit-learn (EconML optional).

**On Kaggle (recommended; GPU T4, 30 GB RAM):**

1. **Code dataset.** Upload `causal_rag_kaggle.zip` as a Kaggle dataset (package, notebooks, results of all runs, README, RESULTS_INDEX, FINAL_SUMMARY). Attach only one copy.
2. **Dry run.** Open `FINAL_v2_0_last_run.ipynb` with `SMOKE = True` (small real sample) to check setup. The setup cell verifies the GPU and code version.
3. **Full run.** Set `SMOKE = False` and run as a Commit (Save & Run All). Takes about 1.5-2 h on a T4 and builds `FINAL_REPORT.html/.md` with figures.
4. **Artifacts.** A run's `artifacts/` folder (cleaned loans, CALF models and scores, indexes, evidence bundles, LLM scores) can be attached to the next notebook from the previous notebook's Output tab or saved as a dataset. Stages skip themselves when their outputs exist; long stages are chunked and resumable.
5. **Keep the run.** Copy the export folder into the dataset's `results/` directory; every new run compares itself with the archived ones.

---

## Live demo

`LIVE_DEMO.ipynb` launches a Gradio app with:

- **Audit tab:** decision, flags, causal DAG and per-clause table for an applicant.
- **What-if sweep tab:** plots P(decline) against a continuous field for several prompt designs, making cliff behaviour visible.

> The live demo has not yet been run on Kaggle with the real model.

---

## Limitations

- Effects are **observational** and depend on the causal graph. Ignorability is untestable (E-values about 1.8-2.1). Funded loans only (selection bias, no reject inference). Clauses are group contrasts, not nudges of a continuous value.
- The four **cutoffs are analyst-defined**; the retrieved policy text states none of them, and the 36% Fannie Mae DTI cutoff binds on only 1.5% of these loans.
- **Small evaluation sample:** 600 applicants with 85 defaults, so AUROC differences below about 0.05 are within noise. The cliff test uses 150 applicants.
- **One LLM** (Qwen2.5-7B, 4-bit) and **one prompt family**; single runs per configuration.
- The cliff test cannot separate cutoff wording from the evidence content that accompanies it.
- **This is a research project, not a lending decision system.**

---

## Future work

- **Dose-response prompts** giving matched-precedent evidence by value band (not by regime), to see whether the residual cliff disappears while keeping the causal content.
- **Larger evaluation** (more defaults) and several LLMs, with fine-tuning or per-system threshold calibration.
- **CALF on genuinely hierarchical covariates** (the paper's NLSY97 setting) to retest the structure hypothesis.
- **Reject inference** and conformal recalibration of CALF's uncertainty; sensitivity analyses beyond E-values.

---

## Author and citation

**Agnes Teresa Davuri** ([@Davuri124](https://github.com/Davuri124)), B.Tech CSE, National Institute of Technology, Warangal.

Built on CALF (Causally-Aware Layered Fabric), the author's own tree-structured model for individual treatment effect estimation.

```bibtex
@misc{davuri2026causalrag,
  author = {Davuri, Agnes Teresa},
  title  = {Causal-RAG: Counterfactual-Audited Retrieval for High-Stakes Lending Decisions},
  year   = {2026},
  note   = {Version 2.0.0}
}
```

## Acknowledgements and data sources

- Lending Club accepted loans via Kaggle (`wordsforthewise/lending-club`)
- Fannie Mae Selling Guide; CFPB / eCFR (12 CFR 1026.43, 12 CFR 1002.6)
- Hillstrom MineThatData e-mail experiment
- Qwen2.5-7B-Instruct; `BAAI/bge-small-en-v1.5`
