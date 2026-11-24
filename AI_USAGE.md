# AI Usage

## Milestone 1

| Tool | Used for |
|---|---|
| Claude (Anthropic) | Starter code for prepare_data.py, train.py, app.py, tests, README/CONTRIBUTION templates |
| TODO | TODO |

All AI-generated code was reviewed, run, and understood by the team before being committed. We adapted it to our dataset and are able to explain every part of it.

## Milestone 2

| Tool | Used for |
|---|---|
| Claude (Anthropic) | Monitoring middleware and `/stats`, latency logging in MLflow, CI workflow (lint, test, build) |
| Claude (Anthropic) | Render deployment settings and finding the cause of the slow build (`OMP_NUM_THREADS=1`) |
| Claude (Anthropic) | DataOps: fetching new books, the three book readers (MNFSR, Punjab CRS, Sindh OCR), yield validation, monthly data workflow, climate updates |
| Claude (Anthropic) | Data versioning (`DATA_VERSIONS.csv`, MLflow, `/version`) |
| Claude (Anthropic) | Drift detection and Discord alert, retraining with a promotion gate |
| Claude (Anthropic) | Blue-green deployment (router, switch, Docker compose) |
| Claude (Anthropic) | Explaining errors and fixing them; drafting README.md, RUNBOOK.md, CONTRIBUTION.md and this file |

**Review.** All AI-generated code and text was reviewed by the team before it was committed, and in M2 every
part was also checked by hand:

- every change went through a pull request with CI (lint, tests, Docker build) passing before merge;
- we ran the tests and the pipeline steps ourselves on both Mac and Windows before committing;
- the book readers were checked on the real PDFs: their rows were compared with the yield data we had already
  checked by hand in M1, and the province / division totals of every book were checked;
- the deployment was checked on the live service (`/health`, `/predict`, `/stats`, `/version`), and rollback was
  tested by following the runbook;
- blue-green was run with Docker: switch, rollback, and a refused switch to a stopped version.

We understand every part of the code and can explain it.
