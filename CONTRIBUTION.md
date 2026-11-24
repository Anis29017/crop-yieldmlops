# Contribution

| Name | ERP |
|---|---|
| Adeena Arif | 29283 |
| Anis Imran | 29017 |
| Saadan Ashraf | 29223 |

## Task table

| Task | Owner | Status |
|---|---|---|
| Repo setup, venv, .gitignore, project skeleton (prepare / train / API / tests) | Anis Imran | Done |
| District yield data: extraction from PBS, provincial CRS / Development Statistics books and MNFSR; checks and corrections; Pakistan master workbook | Adeena Arif | Done |
| District boundaries (HDX) and district name table | Adeena Arif | Done |
| Weather and soil retrieval (NASA POWER, CHIRPS, SoilGrids) and season features | Saadan Ashraf | Done |
| Merge into data/raw/master.csv, prepare_data.py validation | Adeena Arif | Done |
| train.py + MLflow runs + model registry | Adeena Arif | Done |
| FastAPI app.py | Anis Imran | Done |
| Tests (pytest) + ruff | Anis Imran | Done |
| README / AI_USAGE | Anis Imran | Done |

## Milestone 2 task table

| Task | Owner | Status |
|---|---|---|
| Remove the synthetic-data fallback (missing real inputs now stop the pipeline) | Adeena Arif | Done |
| Dockerfile and .dockerignore (model trained in the build, healthcheck, PORT) | Adeena Arif | Done |
| Monitoring: request timing middleware, `/stats`, latency logged in MLflow | Anis Imran | Done |
| CI: lint, test and Docker build jobs (`.github/workflows/ci.yml`) | Anis Imran | Done |
| Render deployment and settings (build-time fix, auto-deploy after CI) | Anis Imran | Done |
| RUNBOOK.md (redeploy, rollback, older image tag, monthly data update) | Adeena Arif | Done |
| Testing the runbook: redeploy and rollback by following it | Saadan Ashraf | Done |
| DataOps: fetching new yield books, readers for all three sources (MNFSR, Punjab CRS, Sindh OCR), monthly workflow, climate updates | Adeena Arif | Done |
| DataOps: validation of the yield data (`validate_yields.py`) and of every book (totals, names) | Adeena Arif | Done |
| Data versioning (`DATA_VERSIONS.csv`, MLflow, `/version`) | Adeena Arif | Done |
| Drift detection (`drift.py`) and Discord alert | Saadan Ashraf | Done |
| Retraining with a promotion gate (`retrain.py`, `retrain.yml`) | Saadan Ashraf | Done |
| Blue-green deployment (router, checked switch, rollback, Docker compose) | Anis Imran | Done |
| First data workflow run, recording the MNFSR books | Anis Imran | Done |
| README, CONTRIBUTION.md, AI_USAGE.md | Adeena Arif | Done |

## Reflections

### Milestone 1

**Adeena Arif:** Working with agricultural datasets taught me how challenging it can be to collect and standardize data from multiple sources. The biggest difficulty was handling inconsistent district names, missing records, and different reporting formats while ensuring the final dataset remained reliable.

**Anis Imran:** This milestone helped me understand how different components of an MLOps pipeline fit together, from model training and MLflow experiment tracking to API development and automated testing. One of the main challenges was integrating everything into a reproducible workflow while resolving dependency, code quality, and repository structure issues.

**Saadan Ashraf:** I learned practical skills like collecting and processing data. In the case of our project, soil and weather data from external sources like NASA POWER and SoilGrids. The challenge was transforming raw data to season specific features and mapping the data to physical locations while ensuring the names are consistent throughout the data sources and ensure model training faces no data issues

### Milestone 2

**Adeena Arif:** I built the Dockerfile, the DataOps pipeline that reads new yield books from three government sources (including a scanned book with OCR), its validation checks, and data versioning. I learned that a reader cannot handle every future book perfectly, so the safety lies in checks that must add up and a person approving every change before it reaches the model, with every model traceable to the exact data it was trained on.

**Anis Imran:** I set up monitoring, CI, the Render deployment (cutting the build from over an hour to about four minutes) and ran blue-green deployment in Docker. I learned how automated checks keep broken code out of main, and how two version-tagged images with a checked switch let a new version go live, or roll back, in one step without risk.

**Saadan Ashraf:** Working on M2 made me realize that getting a good model is only half the job. Keeping it reliable when real world conditions change is essential as well. When I built the drift checker and tested it on the 2022 flood data, seeing how much weather numbers can shift showed me why a static model would fail. Setting up the retraining pipeline also taught me that automation needs safety checks so a worse model never accidentally replaces the live one. Discord alerts was a new thing I learned as I've never used it with other apps especially github.
