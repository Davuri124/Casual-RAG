# v1.3 run: new controls on the same 600 evidence bundles (GPU T4)
Code 1.3.0. Systems: conditions-only control, graded prompt, boundary flips for no_rag / vanilla_rag. LLM stage 295 min.
NOTE: the 'boundary flip' here moves a field to just across the cutoff FROM WHEREVER IT IS (not a +-0.5 nudge), so absolute 'cliff ratios' are confounded by move size; compare systems relatively. Fixed in v1.4 (local cliff test).
Source: user's casual-rag-pipeline.ipynb outputs.
