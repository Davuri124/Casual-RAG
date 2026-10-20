"""
Run history: compare this run against every earlier run stored in the dataset's results/<run_name>/ folders,
and export this run in the same format so it can be added to results/ for future comparisons.

Each run folder may contain: stage08_discrimination.csv, stage08_counterfactual_table.csv, stage08_coverage_risk.json,
RUN_INFO.md  (and any logs / extra tables).
"""
import glob
import json
import os
import shutil
import datetime as dt
import pandas as pd

from . import paths


def previous_runs():
    dirs = {}
    for d in paths.find_input("results", first=False) or []:
        for run in sorted(glob.glob(os.path.join(d, "*", "stage08_discrimination.csv"))):
            dirs[os.path.basename(os.path.dirname(run))] = os.path.dirname(run)
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")       # running outside Kaggle
    for run in sorted(glob.glob(os.path.join(local, "*", "stage08_discrimination.csv"))):
        dirs.setdefault(os.path.basename(os.path.dirname(run)), os.path.dirname(run))
    return dirs


def all_run_dirs():
    """Every run folder under any results/ directory (even those without a discrimination table, e.g. the cliff run)."""
    out = {}
    roots = list(paths.find_input("results", first=False) or []) + [os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")]
    for r in roots:
        if os.path.isdir(r):
            for d in sorted(os.listdir(r)):
                if os.path.isdir(os.path.join(r, d)):
                    out.setdefault(d, os.path.join(r, d))
    return out


def _read(d, name, kind):
    f = os.path.join(d, name)
    if not os.path.exists(f):
        return None
    return pd.read_csv(f) if kind == "csv" else json.load(open(f))


def all_runs(current_name, disc, cft, cov):
    runs = previous_runs()
    D, C, V = [], [], []
    for name, d in runs.items():
        x = _read(d, "stage08_discrimination.csv", "csv")
        if x is not None: D.append(x.assign(run=name))
        x = _read(d, "stage08_counterfactual_table.csv", "csv")
        if x is not None: C.append(x.assign(run=name))
        x = _read(d, "stage08_coverage_risk.json", "json")
        if x is not None: V.append({**x, "run": name})
    D.append(disc.assign(run=current_name)); C.append(cft.assign(run=current_name)); V.append({**cov, "run": current_name})
    out = dict(discrimination=pd.concat(D, ignore_index=True), counterfactual=pd.concat(C, ignore_index=True), coverage=pd.DataFrame(V))
    for k, v in out.items():
        v.to_csv(paths.art(f"all_runs_{k}.csv"), index=False)
    return out


def export_run(name, disc, cft, cov, config, extra_files=()):
    """Writes ART/results_export/<name>/ ... copy this folder into the dataset's results/ folder to keep the run."""
    out = paths.ART / "results_export" / name
    out.mkdir(parents=True, exist_ok=True)
    disc.to_csv(out / "stage08_discrimination.csv", index=False)
    cft.to_csv(out / "stage08_counterfactual_table.csv", index=False)
    json.dump(cov, open(out / "stage08_coverage_risk.json", "w"), indent=1)
    for f in extra_files:
        if os.path.exists(f):
            shutil.copy(f, out / os.path.basename(f))
    (out / "RUN_INFO.md").write_text(f"# {name}\nExported {dt.datetime.now():%Y-%m-%d %H:%M}\n\n```json\n"
                                     + json.dumps(config, indent=1, default=str) + "\n```\n")
    return out
