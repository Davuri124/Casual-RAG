"""Artifact/IO conventions. Every stage reads/writes ART so stages can be chained across Kaggle sessions."""
import os, shutil, glob, pathlib

ON_KAGGLE = os.path.exists("/kaggle/input")
ART = pathlib.Path("/kaggle/working/artifacts" if ON_KAGGLE else os.environ.get("CRAG_ART", "./artifacts"))
ART.mkdir(parents=True, exist_ok=True)
INPUT = pathlib.Path("/kaggle/input")

KNOWN = {  # artifact files that later stages need
    "lc_clean.parquet", "censoring_report.csv", "graph.json", "adjustment_sets.json", "graph.png",
    "policy_chunks.parquet", "policy_emb.npy", "case_cards.parquet", "case_emb.npy",
    "featurizers.pkl", "calf_scores.parquet", "calf_models.pt", "calf_results.csv",
    "rct_results.csv", "refutations.csv", "evidence.pkl", "eval_sample.parquet",
    "llm_scores.parquet", "cf_audit.parquet", "decisions.jsonl", "metrics.json", "results_table.csv",
    "gbm_scores.parquet", "llm_scores_v20.parquet", "cliff_pairs_v20.parquet", "cliff_pairs.parquet", "llm_scores_v13.parquet", "cf_audit_v13.parquet", "counterfactual_table.csv", "coverage_risk.json", "pairwise_table.csv",
}


def art(name):
    return ART / name


def have(*names):
    return all((ART / n).exists() for n in names)


def find_input(pattern, first=True):
    hits = glob.glob(str(INPUT / "**" / pattern), recursive=True)
    if not hits:
        return None if first else []
    return hits[0] if first else hits


def restore_artifacts(verbose=True):
    """
    Copy known artifacts from ANY attached dataset / notebook output into ART, wherever they sit in /kaggle/input
    (Kaggle mounts at varying depths, e.g. /kaggle/input/datasets/<user>/<slug>/artifacts/...). Files under a `results/`
    folder (archived run tables) are never treated as artifacts.
    """
    if not INPUT.exists():
        return
    n = 0
    ok = lambda h: "/results/" not in h.replace("\\", "/") and "results_export" not in h
    for name in sorted(KNOWN):
        if (ART / name).exists():
            continue
        hits = [h for h in glob.glob(str(INPUT / "**" / name), recursive=True) if os.path.isfile(h) and ok(h)]
        if hits:
            shutil.copy2(hits[0], ART / name); n += 1
    for sub in ("audits", "policy_docs", "llm_parts", "llm_parts_v13", "cliff_parts", "llm_parts_v20", "cliff_parts_v20"):
        if (ART / sub).exists():
            continue
        hits = [h for h in glob.glob(str(INPUT / "**" / sub), recursive=True) if os.path.isdir(h) and ok(h + "/")]
        if hits:
            shutil.copytree(hits[0], ART / sub); n += 1
    if verbose:
        print(f"restored {n} artifact(s) -> {ART}")
