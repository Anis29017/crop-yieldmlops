"""Data versioning: which exact data every model was trained on.

A data version is the fingerprint (first 12 characters of the SHA-256) of data/processed.csv,
the table the models train on. The same data always gives the same version, and any change -
a new book, a corrected number, a new climate season - gives a new one. processed.csv is built
from the input files below by build_yield_labels -> build_master -> prepare_data, and that
build is repeatable (the same inputs give the same file), so a version also pins its inputs.

Three places use it:
  1. data/DATA_VERSIONS.csv (in git): one line per data version - date, version, rows,
     newest year per crop, the fingerprint of every input file, and why it changed.
     The monthly DataOps workflow adds a line in its pull request when the data changes,
     so every data version is reviewed and kept in the git history.
  2. MLflow: every training run logs the data version, the git commit and the row count,
     plus the dataset itself (mlflow.log_input) and data_version.json with all fingerprints.
  3. models/best_run.json and the API's /version endpoint: the live model says which data
     version and code it was trained on.

To get the data of an old version back: find its line in DATA_VERSIONS.csv, check out the
commit that added that line (git log -S <version> data/DATA_VERSIONS.csv) and rebuild.

Usage:
  python -m src.data_version                      # print the current data version
  python -m src.data_version --record "reason"    # add a line to DATA_VERSIONS.csv if the data changed
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

PROCESSED = "data/processed.csv"
VERSIONS = "data/DATA_VERSIONS.csv"
INPUTS = [
    "data/raw/Pakistan_Master_Yield_1981-2025.xlsx",
    "data/sources/new_yields/*.csv",
    "data/interim/climate_features.csv",
    "data/interim/soil_features.csv",
    "data/interim/district_points.csv",
    "data/raw/district_name_table.csv",
]
CROPS = ["wheat", "rice", "cotton", "sugarcane"]
COLUMNS = ["date", "data_version", "rows", "first_year"] + [f"newest_{c}" for c in CROPS] + \
          ["new_book_files", "inputs_sha256", "reason"]


def sha256(path):
    """Fingerprint of a file. CSV files are read with Windows line endings (CRLF) turned into LF,
    so the same data gives the same version on Windows, Mac and Linux."""
    with open(path, "rb") as f:
        content = f.read()
    if path.endswith(".csv"):
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def input_files(patterns=INPUTS):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p)) if "*" in p else ([p] if os.path.exists(p) else [])
    return files


def git_commit():
    """The code version: from the CI / Render environment, else from git, else 'unknown'."""
    for var in ("GITHUB_SHA", "RENDER_GIT_COMMIT"):
        if os.environ.get(var):
            return os.environ[var][:12]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()[:12] or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def describe(processed=PROCESSED, patterns=INPUTS):
    """Everything that identifies the data: version, size, years and the fingerprint of every input."""
    df = pd.read_csv(processed)
    files = {f: sha256(f) for f in input_files(patterns)}
    newest = df.groupby("crop")["year"].max()
    return {
        "data_version": sha256(processed)[:12],
        "processed_file": processed,
        "processed_sha256": sha256(processed),
        "rows": int(len(df)),
        "first_year": int(df["year"].min()),
        "newest_year": {c: int(newest[c]) for c in CROPS if c in newest},
        "rows_per_crop": {c: int(n) for c, n in df["crop"].value_counts().sort_index().items()},
        "new_book_files": sum(1 for f in files if "new_yields" in f),
        "inputs": files,
        "inputs_sha256": hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode())
        .hexdigest()[:12],
        "git_commit": git_commit(),
    }


def log_to_mlflow(info, df=None, target=None):
    """Inside an active MLflow run: the data version as params and tag, the fingerprints as a
    file, and the dataset itself as the run's training input."""
    import mlflow

    mlflow.log_params({"data_version": info["data_version"], "data_rows": info["rows"],
                       "data_newest_year": max(info["newest_year"].values()), "git_commit": info["git_commit"]})
    mlflow.set_tag("data_version", info["data_version"])
    mlflow.log_dict(info, "data_version.json")
    if df is not None:
        dataset = mlflow.data.from_pandas(df, source=info["processed_file"], targets=target,
                                          name=f"processed@{info['data_version']}")
        mlflow.log_input(dataset, context="training")


def summary_fields(info):
    """The fields copied into models/best_run.json (and shown by the API's /version)."""
    return {"data_version": info["data_version"], "data_rows": info["rows"],
            "data_newest_year": info["newest_year"], "data_inputs_sha256": info["inputs_sha256"],
            "git_commit": info["git_commit"]}


def record(reason, processed=PROCESSED, versions=VERSIONS, patterns=INPUTS, today=None):
    """Add a line to DATA_VERSIONS.csv unless the newest line already has this version. Returns the line or None."""
    info = describe(processed, patterns)
    old = pd.read_csv(versions) if os.path.exists(versions) else pd.DataFrame(columns=COLUMNS)
    if len(old) and old.iloc[-1]["data_version"] == info["data_version"]:
        return None
    line = {"date": (today or datetime.now(timezone.utc).date()).isoformat(), "data_version": info["data_version"],
            "rows": info["rows"], "first_year": info["first_year"],
            **{f"newest_{c}": info["newest_year"].get(c) for c in CROPS},
            "new_book_files": info["new_book_files"], "inputs_sha256": info["inputs_sha256"], "reason": reason}
    out = pd.concat([old, pd.DataFrame([line])], ignore_index=True)[COLUMNS]
    out.to_csv(versions, index=False)
    return line


def main(argv=None):
    ap = argparse.ArgumentParser(description="Show or record the data version")
    ap.add_argument("--record", metavar="REASON", help="add a line to data/DATA_VERSIONS.csv if the data changed")
    ap.add_argument("--report", help="markdown file to append the result to (e.g. the pull request text)")
    args = ap.parse_args(argv)
    if args.record is None:
        print(json.dumps(describe(), indent=2))
        return 0
    line = record(args.record)
    text = ("## Data version\n\n" + (f"New data version **{line['data_version']}** ({line['rows']} rows; "
            f"newest: " + ", ".join(f"{c} {line[f'newest_{c}']}" for c in CROPS) + ")."
            if line else "The data did not change: same data version as before.") + "\n")
    print(text)
    if args.report:
        with open(args.report, "a") as f:
            f.write("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
