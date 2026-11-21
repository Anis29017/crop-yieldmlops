"""One command for the model part of the pipeline:
yield labels -> merge -> prepare data -> train -> tests.

    python -m src.run_pipeline

Steps 1-5 (districts, NASA POWER, CHIRPS, SoilGrids, climate features) download
data and are run separately; see the README. This script uses their outputs in
data/interim/. If any input is missing, it stops with an error: no data is
generated or substituted.
"""

import os
import subprocess
import sys

from src import build_master, build_yield_labels
from src.prepare_data import prepare_data
from src.train import train

NEEDED = [
    build_yield_labels.YIELD_XLSX,
    build_yield_labels.POINTS_PATH,
    build_master.CLIMATE_PATH,
    build_master.SOIL_PATH,
]


def main():
    missing = [p for p in NEEDED if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}. They must come from the real data sources (see README).")
    build_yield_labels.main()
    build_master.main()
    prepare_data()
    summary = train()
    code = subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"])
    if code != 0:
        sys.exit(code)
    if not summary["beats_baseline"]:
        print("WARNING: best model does not beat the last-year baseline")
        sys.exit(1)
    print("Pipeline OK: data prepared, model trained and registered, tests pass.")


if __name__ == "__main__":
    main()
