# Crop yield prediction API (Milestone 2)
#
# Build:  docker build -t crop-yield:v1 .
# Run:    docker run -p 8000:8000 crop-yield:v1
# Then open http://localhost:8000/docs
#
# The model is NOT in the repository: it is trained here, during the build,
# from the real data files committed in data/.

# Python 3.12: xgboost 3.4.1 in requirements.txt needs Python >= 3.12
# (the same version the GitHub Actions CI uses).
FROM python:3.12-slim

# OMP_NUM_THREADS=1: XGBoost/LightGBM start one thread per CPU they can see. On shared
# build machines (Render) that many threads fight over a small CPU share and made the
# build take over an hour; one thread gives the same results in seconds on this data.
# GIT_PYTHON_REFRESH=quiet: the image has no git, so MLflow cannot record the commit;
# this only silences that warning.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    LOKY_MAX_CPU_COUNT=1 \
    GIT_PYTHON_REFRESH=quiet

# LightGBM and XGBoost need the OpenMP library, which the slim image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Dependencies first, so Docker reuses this layer when only the code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Source code and the real input data (no generated data anywhere).
COPY src/ src/
COPY data/raw/Pakistan_Master_Yield_1981-2025.xlsx data/raw/district_name_table.csv data/raw/
COPY data/interim/district_points.csv data/interim/climate_features.csv data/interim/soil_features.csv data/interim/
# Rows from new books, added by the DataOps pipeline through reviewed pull requests.
COPY data/sources/new_yields/ data/sources/new_yields/

# The code version recorded with the model (src/data_version.py); the image has no git.
# CI passes GITHUB_SHA; Render passes its environment variables, RENDER_GIT_COMMIT among them,
# as build arguments. If neither is given, the model records "unknown".
ARG GITHUB_SHA
ARG RENDER_GIT_COMMIT

# 3. Build the training table and train the model. Any missing input stops the build.
RUN python -m src.build_yield_labels \
    && python -m src.build_master \
    && python -m src.prepare_data \
    && python -m src.train \
    && test -f models/model.pkl

EXPOSE 8000

# Docker marks the container "healthy" once /health answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '8000') + '/health', timeout=4)" || exit 1

# Listen on all interfaces; hosting platforms set PORT, locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
