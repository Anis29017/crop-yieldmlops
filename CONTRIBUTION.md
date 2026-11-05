# Contribution

| Name | ERP |
|---|---|
| Adeena Arif | 29283 |
| Anis Imran | 29017 |
| Saadan Ashraf | 29223 |

## Milestone 1 task table

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

## Reflections

**Adeena Arif:** Working with agricultural datasets taught me how challenging it can be to collect and standardize data from multiple sources. The biggest difficulty was handling inconsistent district names, missing records, and different reporting formats while ensuring the final dataset remained reliable.

**Anis Imran:** This milestone helped me understand how different components of an MLOps pipeline fit together, from model training and MLflow experiment tracking to API development and automated testing. One of the main challenges was integrating everything into a reproducible workflow while resolving dependency, code quality, and repository structure issues.

**Saadan Ashraf:** I learned practical skills like collecting and processing data. In the case of our project, soil and weather data from external sources like NASA POWER and SoilGrids. The challenge was transforming raw data to season specific features and mapping the data to physical locations while ensuring the names are consistent throughout the data sources and ensure model training faces no data issues
