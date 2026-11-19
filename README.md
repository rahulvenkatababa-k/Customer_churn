<div align="center">

<img src="images/Customer%20Churn%20MLOps%20Dashboard.png" alt="Customer Churn MLOps" width="180"/>

# Customer Churn MLOps

**An end-to-end machine learning system that predicts telecom customer churn — from versioned data and tracked experiments to a containerised, Kubernetes-deployed, monitored prediction service.**

![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2?logo=mlflow&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-Orchestration-070E10?logo=prefect&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containers-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Helm-326CE5?logo=kubernetes&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [ML Workflow](#5-ml-workflow)
6. [Serving Layer](#6-serving-layer)
7. [Monitoring](#7-monitoring)
8. [Setup](#8-setup)
9. [Running the Project](#9-running-the-project)
10. [Testing](#10-testing)
11. [CI/CD](#11-cicd)
12. [Deployment](#12-deployment)
13. [Configuration Reference](#13-configuration-reference)
14. [Current Scope and Known Gaps](#14-current-scope-and-known-gaps)

---

## 1. Overview

Customer churn is a classic imbalanced-classification problem: only a minority of customers leave, so accuracy alone is a poor guide. This project treats churn prediction as a **production system**, not a notebook:

- **Data** is read from a versioned PostgreSQL table, snapshotted with **DVC**, and validated against a strict **Pandera** schema before any training happens.
- **Training** compares a logistic-regression baseline against AdaBoost, XGBoost and CatBoost, using class weighting, SMOTE / SMOTEENN resampling, `GridSearchCV` and **Optuna**, all tracked in **MLflow**.
- **Model selection** is done on a validation split (F1, then recall, then average precision); the held-out test set is touched only once, for the chosen model.
- **Promotion** registers the winner in the MLflow Model Registry and assigns the `champion` alias.
- **Serving** is a **FastAPI** service that always resolves `champion` to an exact model version, logs every prediction to PostgreSQL, and exposes Prometheus metrics. A **Streamlit** app provides a UI on top.
- **Operations** are covered by Docker, GitHub Actions (test, build, scan, publish, deploy), Helm on Kubernetes, Prometheus/Grafana dashboards and **Evidently** data-drift checks on live traffic.

**Dataset:** a telecom customer-churn dataset — 7,043 customers, 21 columns, with 1,869 churners (`Yes`) and 5,174 non-churners (`No`), tracked at `data/versioned/customer_churn.csv`.

---

## 2. Architecture

```mermaid
flowchart LR
    subgraph DATA["Data layer"]
        PG[("PostgreSQL<br/>customer_churn_vN")]
        DVC["DVC snapshot<br/>(DagsHub remote)"]
    end

    subgraph TRAIN["Training pipeline (Prefect flow)"]
        ING[Ingestion] --> VAL[Validation<br/>Pandera]
        VAL --> TRF[Transformation]
        TRF --> FE[Feature Engineering<br/>70/15/15 split]
        FE --> TRN[Model Trainer<br/>Grid + Optuna]
        TRN --> EVA[Model Evaluation]
        EVA --> PUSH[Model Pusher]
    end

    subgraph REG["MLflow (DagsHub)"]
        EXP[Experiments & metrics]
        MR[("Model Registry<br/>alias: champion")]
    end

    subgraph SERVE["Serving"]
        API["FastAPI<br/>/predict /health /metrics"]
        UI[Streamlit UI]
    end

    subgraph OBS["Observability"]
        PROM[Prometheus] --> GRAF[Grafana]
        EVI["Evidently<br/>drift report"]
    end

    PG --> ING
    ING --> DVC
    TRN --> EXP
    EVA --> MR
    PUSH -->|set alias| MR
    PUSH -->|baseline row| PG
    MR -->|load champion| API
    UI -->|HTTP| API
    API -->|prediction_logs| PG
    API -->|/metrics| PROM
    PG -->|recent predictions| EVI
    PG -->|training reference| EVI
```

**How the pieces fit:** the training pipeline reads versioned data, produces a registered model and records a *monitoring baseline* (the training feature set for that model version). The API serves whatever version holds the `champion` alias and logs each request. The drift monitor later compares the logged production features against the baseline for that exact version.

---

## 3. Tech Stack

| Area | Technologies |
|---|---|
| **Language** | Python 3.12 (containers) / 3.13 (CI) |
| **Data & validation** | pandas, NumPy, SQLAlchemy, psycopg2, **Pandera**, PostgreSQL |
| **Data versioning** | **DVC** with a DagsHub remote |
| **Modelling** | scikit-learn, imbalanced-learn (SMOTE, SMOTEENN), **XGBoost**, **CatBoost**, **Optuna** |
| **Experiment tracking & registry** | **MLflow** (hosted on DagsHub), model aliases |
| **Orchestration** | **Prefect** (flows, tasks, retries, timeouts) |
| **API & UI** | **FastAPI**, Uvicorn, Pydantic, **Streamlit** |
| **Containers** | Docker, Docker Compose, GitHub Container Registry (GHCR) |
| **Kubernetes** | Raw manifests, **Helm** chart, NGINX Ingress |
| **CI/CD** | **GitHub Actions**, Trivy image scanning, self-hosted runner |
| **Monitoring** | **Prometheus**, **Grafana** (kube-prometheus-stack), `prometheus-fastapi-instrumentator`, **Evidently** |
| **Testing** | pytest |

---

## 4. Project Structure

```text
Customer_churn/
├── main.py                         # Runs the training pipeline directly
├── streamlit_app.py                # Streamlit front end
├── src/customer_churn/
│   ├── app.py                      # FastAPI application
│   ├── components/
│   │   ├── data_ingestion.py       # Latest customer_churn_vN table → CSV + DVC snapshot
│   │   ├── data_validation.py      # Pandera schema check + JSON report
│   │   ├── data_transformation.py  # TotalCharges cleaning
│   │   ├── feature_engineering.py  # Target encoding + stratified train/val/test split
│   │   ├── model_trainer.py        # Baseline, candidates, GridSearchCV, Optuna
│   │   ├── model_evaluation.py     # Validation ranking, test evaluation, registration
│   │   ├── model_pusher.py         # `champion` alias + monitoring baseline
│   │   ├── prediction.py           # Registry-aware inference
│   │   ├── prediction_logger.py    # Writes predictions to PostgreSQL
│   │   ├── production_data_reader.py
│   │   └── production_drift_monitor.py
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates stages inside one MLflow run
│   │   ├── prefect_pipeline.py     # Prefect flow (retries / timeouts per task)
│   │   └── monitoring_pipeline.py  # Prefect flow for production drift check
│   ├── entity/                     # Config + artifact dataclasses, Pandera schema
│   ├── constants/                  # Pipeline constants
│   ├── configuration/              # Environment-based configuration
│   ├── utils/                      # I/O helpers, MLflow helpers
│   ├── logging/                    # File + console logger
│   └── exception/                  # Custom exception with context
├── evidently_scripts/              # CLI for ad-hoc drift reports
├── tests/                          # pytest suite
├── notebooks/                      # EDA and model-building experiments
├── data/versioned/                 # DVC-tracked dataset snapshot
├── reports/evidently/              # Generated drift reports (HTML)
├── k8s/                            # Raw Kubernetes manifests
├── helm/customer-churn/            # Helm chart (API, Streamlit, Ingress, ServiceMonitor)
├── monitoring/                     # kube-prometheus-stack values + Grafana dashboard JSON
├── terraform/                      # Module scaffold (files currently empty)
├── .github/workflows/              # ci.yml, cd.yml, runner-check.yml
├── Dockerfile                      # API image
├── Dockerfile.streamlit            # Streamlit image
├── docker-compose.yml              # Build-from-source stack
├── docker-compose-ghcr.yml         # Stack using published GHCR images
└── cmds.md                         # Kubernetes / Helm / Prometheus runbook
```

---

## 5. ML Workflow

The training pipeline has seven stages. Each stage consumes the previous stage's **artifact** (a typed dataclass), and all outputs for a run are written under `artifacts/<timestamp>/`.

| # | Stage | What it does |
|---|---|---|
| 1 | **Data Ingestion** | Connects to PostgreSQL, discovers the highest-numbered `customer_churn_vN` table in `public`, reads it, sanity-checks it (non-empty, unique columns), and writes both a per-run CSV and a stable snapshot at `data/versioned/customer_churn.csv` for DVC. |
| 2 | **Data Validation** | Validates the data with a **strict Pandera schema** (allowed categorical values, `SeniorCitizen ∈ {0,1}`, non-negative `tenure` / `MonthlyCharges`, unique non-null `customerID`, `Churn ∈ {Yes, No}`). Writes `validation_report.json`; if validation fails, no validated file is produced and the next stage stops with an error. |
| 3 | **Data Transformation** | Converts `TotalCharges` to numeric (blank strings become missing) and fills the missing values with `0`. |
| 4 | **Feature Engineering** | Maps `Churn` to 0/1, drops `customerID`, and creates a **stratified 70 / 15 / 15 train / validation / test split** (seed 42). |
| 5 | **Model Training** | Fits a logistic-regression **baseline**, then candidate models and tuned experiments (below). |
| 6 | **Model Evaluation** | Scores every model on the **validation** set, ranks them, selects one, then evaluates **only that model** on the held-out test set. |
| 7 | **Model Pusher** | Registers the model, sets the `champion` alias, and stores a monitoring baseline row in PostgreSQL. |

### Modelling approach

All models are scikit-learn / imbalanced-learn `Pipeline`s that bundle preprocessing with the estimator — `StandardScaler` for numeric columns, one-hot encoding (unknown categories ignored) for categoricals — so there is **no preprocessing leakage across cross-validation folds** and the exact same transformation is applied at inference.

| Group | Models |
|---|---|
| Baseline | Logistic Regression |
| Candidates | AdaBoost, AdaBoost with balanced sample weights, XGBoost, CatBoost |
| `GridSearchCV` (5-fold stratified, scoring = F1) | AdaBoost + SMOTE, AdaBoost + SMOTEENN, XGBoost + `scale_pos_weight`, XGBoost + SMOTE, CatBoost + class weights |
| Optuna (TPE sampler, 20 trials each, 5-fold stratified F1) | AdaBoost + SMOTE, AdaBoost + SMOTEENN |

### Imbalance handling and selection

- Class imbalance is addressed three ways: **sample / class weights**, **SMOTE**, and **SMOTEENN**.
- Models are ranked on validation by **F1**, with **recall** and then **average precision** as tie-breakers. An F2 score is also computed for the summary table.
- Reported metrics: accuracy, precision, recall, F1, ROC-AUC, average precision, and confusion matrices (PNG).

### Experiment tracking and registry

- One parent MLflow run (`customer-churn-pipeline`) wraps the whole pipeline; each tuning experiment is logged as a **nested run** with its best parameters and cross-validated F1.
- The selected model is logged with an inferred **signature** and **input example** and registered as `customer_churn_model`.
- The pusher assigns the **`champion`** alias to that version and writes `(model_name, model_version, reference_data_path)` to `model_monitoring_baselines`, linking each version to the data it was trained on.

### Orchestration

`src/customer_churn/pipeline/prefect_pipeline.py` wraps every stage in a Prefect task with explicit timeouts, and adds retries (3 × 30 s) on the stages that touch external systems — **ingestion** and **model pusher**. `main.py` runs the same pipeline without Prefect.

---

## 6. Serving Layer

### FastAPI (`src/customer_churn/app.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service banner |
| `/health` | GET | Liveness / readiness probe target |
| `/predict` | POST | Returns `Prediction` (0/1) and `Probability` of churn |
| `/metrics` | GET | Prometheus metrics |

Behaviour worth noting:

- The model URI is `models:/customer_churn_model@champion`. On first use the alias is resolved to an **exact version number**, and inference is pinned to that version.
- Each prediction is persisted to the `prediction_logs` table (features, prediction, probability, model version, UTC timestamp) in a **background task**, so logging does not add latency to the response.
- Request bodies are validated by a Pydantic model covering all 19 customer features.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

Response shape (values depend on the current `champion` model):

```json
{ "Prediction": <0 or 1>, "Probability": <churn probability between 0 and 1> }
```

### Streamlit (`streamlit_app.py`)

A form-based UI that collects the 19 customer attributes, calls `PREDICTION_API_URL` (default `http://127.0.0.1:8000/predict`), and shows the churn decision and probability.

---

## 7. Monitoring

### Service monitoring — Prometheus and Grafana

- The API exposes request count and latency metrics via `prometheus-fastapi-instrumentator`.
- The Helm chart ships a **ServiceMonitor** (labelled `release: monitoring`) that scrapes `/metrics` every 30 s.
- `monitoring/monitoring-values.yaml` configures **kube-prometheus-stack** (Prometheus, Grafana with persistence, Alertmanager, kube-state-metrics, node-exporter).
- `monitoring/grafana/customer-churn-dashboard.json` is an exportable Grafana dashboard with seven panels: **API request rate, HTTP response rate by status, availability, p95 latency, error rate (%), pod CPU usage, pod memory usage**.

### Data-drift monitoring — Evidently

Two entry points use the Evidently `DataDriftPreset`:

| Mode | Command | Compares |
|---|---|---|
| **Offline** | `python -m evidently_scripts.evidently_monitoring` | `X_train.csv` vs `X_test.csv` from the latest artifact run (or any two CSVs via `--reference` / `--current`) |
| **Production** | `python -m src.customer_churn.pipeline.monitoring_pipeline` | Training reference for the live `champion` version vs the most recent 1,000 logged predictions for that version |

The production check resolves the `champion` version through MLflow, fetches its baseline from `model_monitoring_baselines`, verifies that the feature schemas match, and requires at least **100** production rows before generating `reports/evidently/production_drift_report.html`; otherwise it reports `insufficient_data`.

---

## 8. Setup

### Prerequisites

- Python 3.12+ and `pip`
- PostgreSQL (local or hosted)
- An MLflow tracking server (the project is configured for DagsHub-hosted MLflow, but any server works)
- Docker, and for Kubernetes: `kubectl`, Helm and a cluster (Docker Desktop Kubernetes was used)
- Optional: DVC, and the Prefect CLI for UI-based runs

### 1. Clone and install

```bash
git clone https://github.com/rahulvenkatababa-k/Customer_churn.git
cd Customer_churn

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Fill in `.env` (full reference in [section 13](#13-configuration-reference)). **Never commit `.env`.**

### 3. Prepare PostgreSQL

The ingestion stage reads the highest-numbered table matching `customer_churn_v<N>` in the `public` schema, so load the dataset into a table with that name. For example, with pandas:

```python
import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("data/versioned/customer_churn.csv")
engine = create_engine("<your POSTGRES_SQL_URI>")
df.to_sql("customer_churn_v1", engine, if_exists="replace", index=False)
```

The pipeline and API also use two operational tables. Their DDL is not shipped in the repository; the following is a schema consistent with what the code reads and writes:

```sql
CREATE TABLE IF NOT EXISTS prediction_logs (
    id                 BIGSERIAL PRIMARY KEY,
    predicted_at       TIMESTAMPTZ NOT NULL,
    gender             TEXT,
    "SeniorCitizen"    INTEGER,
    "Partner"          TEXT,
    "Dependents"       TEXT,
    tenure             INTEGER,
    "PhoneService"     TEXT,
    "MultipleLines"    TEXT,
    "InternetService"  TEXT,
    "OnlineSecurity"   TEXT,
    "OnlineBackup"     TEXT,
    "DeviceProtection" TEXT,
    "TechSupport"      TEXT,
    "StreamingTV"      TEXT,
    "StreamingMovies"  TEXT,
    "Contract"         TEXT,
    "PaperlessBilling" TEXT,
    "PaymentMethod"    TEXT,
    "MonthlyCharges"   DOUBLE PRECISION,
    "TotalCharges"     DOUBLE PRECISION,
    prediction         INTEGER,
    probability        DOUBLE PRECISION,
    model_version      TEXT
);

CREATE TABLE IF NOT EXISTS model_monitoring_baselines (
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    reference_data_path TEXT NOT NULL,
    PRIMARY KEY (model_name, model_version)
);
```

### 4. (Optional) Pull versioned data with DVC

```bash
dvc pull        # remote: DagsHub (see .dvc/config)
```

---

## 9. Running the Project

All commands are run from the repository root.

### Train the model

```bash
# Plain Python
python main.py

# As a Prefect flow (retries and timeouts per stage)
python -m src.customer_churn.pipeline.prefect_pipeline
```

To watch runs in the Prefect UI, start a local server first (`prefect server start`) and keep `PREFECT_API_URL=http://127.0.0.1:4200/api` in `.env`. Experiments, metrics and the registered model appear in your MLflow UI.

### Serve the model

```bash
# API  → http://localhost:8000  (interactive docs at /docs)
uvicorn src.customer_churn.app:app --host 0.0.0.0 --port 8000

# Streamlit UI  → http://localhost:8501   (in a second terminal)
streamlit run streamlit_app.py
```

The API requires a registered model with the `champion` alias, so run the training pipeline at least once first.

### Run drift checks

```bash
python -m evidently_scripts.evidently_monitoring                  # offline: train vs test
python -m src.customer_churn.pipeline.monitoring_pipeline         # production vs training baseline
```

### Run with Docker Compose

```bash
# Build locally
docker compose up --build

# Or use the published GHCR images
docker compose -f docker-compose-ghcr.yml up
```

- The Streamlit UI is published on **http://localhost:8501**.
- The API container loads its settings from `.env`, has a health check on `/health`, and runs with CPU and memory limits. Its port is not published to the host by default; add `ports: ["8000:8000"]` to the `api` service to call it directly.

---

## 10. Testing

```bash
pytest -v tests/
```

The suite contains **58 test functions** covering data ingestion, validation, transformation, feature engineering, model training, model evaluation, the model pusher, and the Evidently drift report. `pytest` is configured in `pyproject.toml` (`testpaths = ["tests"]`).

> Several modules read configuration when they are imported, so the variables in [section 13](#13-configuration-reference) must be set (via `.env` locally, or secrets in CI) before running the tests.

---

## 11. CI/CD

### Continuous Integration — `.github/workflows/ci.yml`

Triggered on every **push to `main`**.

```text
validate-and-build                          publish-images (needs: validate-and-build)
├─ checkout                                 ├─ log in to GHCR
├─ set up Python 3.13                       ├─ build API + Streamlit images
├─ pip install -r requirements.txt          ├─ Trivy scan (API, Streamlit) → SARIF
├─ pytest -v tests/                         ├─ upload SARIF to GitHub code scanning + artifacts
├─ docker build API image                   └─ push images tagged :<commit-sha> and :latest
└─ docker build Streamlit image
```

- Images are published to `ghcr.io/rahulvenkatababa-k/customer-churn-api` and `.../customer-churn-streamlit`.
- Trivy scans for **CRITICAL** OS and library vulnerabilities (unfixed ones ignored). The Streamlit scan is configured to fail the job; the API scan currently reports without failing.
- Pipeline credentials (MLflow, PostgreSQL) come from GitHub **environment secrets**.

### Continuous Deployment — `.github/workflows/cd.yml`

Runs when **CI completes successfully on `main`**, on a **self-hosted Windows runner** that has access to the local Kubernetes cluster:

1. Check out the exact commit that passed CI.
2. `helm upgrade --install customer-churn helm/customer-churn --wait --timeout 5m`, setting both image tags to the **commit SHA** (immutable, traceable deployments).
3. `kubectl rollout status` for the API and Streamlit deployments.
4. List running pods.

`runner-check.yml` is a manual (`workflow_dispatch`) workflow that verifies the runner can reach `kubectl` and Helm.

---

## 12. Deployment

### Kubernetes with Helm (recommended)

The chart in `helm/customer-churn/` deploys:

| Resource | Details |
|---|---|
| API Deployment | 2 replicas, readiness and liveness probes on `/health`, CPU / memory requests and limits |
| Streamlit Deployment | 1 replica, probes on `/_stcore/health`, CPU / memory requests and limits |
| Services | `ClusterIP` for API (8000) and Streamlit (8501); the API port is named `http` |
| Ingress | NGINX, routes `/` to Streamlit (toggle with `ingress.enabled`) |
| ServiceMonitor | Prometheus scrape of `/metrics` every 30 s |
| ConfigMaps | `MLFLOW_EXPERIMENT_NAME`, `MODEL_NAME`, `PREDICTION_API_URL` |

The API reads its credentials from a Secret named **`customer-churn-secrets`**, which the chart does **not** create. Provide it separately (keys: `POSTGRES_SQL_URI`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`):

```bash
kubectl create secret generic customer-churn-secrets \
  --from-literal=POSTGRES_SQL_URI='<value>' \
  --from-literal=MLFLOW_TRACKING_URI='<value>' \
  --from-literal=MLFLOW_TRACKING_USERNAME='<value>' \
  --from-literal=MLFLOW_TRACKING_PASSWORD='<value>'
```

```bash
# 1. NGINX ingress controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --set controller.hostPort.enabled=true

# 2. Application
helm lint helm/customer-churn
helm upgrade --install customer-churn helm/customer-churn \
  --namespace default --wait --timeout 5m \
  --set-string api.image.tag=<tag> \
  --set-string streamlit.image.tag=<tag>

# 3. Verify
kubectl get pods -n default
kubectl rollout status deployment/customer-churn-api -n default
```

### Kubernetes with raw manifests

`k8s/` contains equivalent plain manifests (ConfigMaps, Secret, Deployments, Services, Ingress):

```bash
kubectl apply -f k8s/
```

### Monitoring stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values monitoring/monitoring-values.yaml \
  --version 92.3.0 --wait --timeout 10m

# Access Grafana and Prometheus locally
kubectl port-forward -n monitoring service/monitoring-grafana 3000:80
kubectl port-forward -n monitoring service/monitoring-kube-prometheus-prometheus 9090:9090
```

Import `monitoring/grafana/customer-churn-dashboard.json` into Grafana. See [`cmds.md`](cmds.md) for the full runbook, including PromQL queries used and verification steps.

---

## 13. Configuration Reference

| Variable | Required | Used by | Description |
|---|---|---|---|
| `POSTGRES_SQL_URI` | Yes | Pipeline, API | SQLAlchemy URL, e.g. `postgresql+psycopg2://user:pass@host:5432/dbname` |
| `MLFLOW_TRACKING_URI` | Yes | Pipeline, API | MLflow tracking server URL |
| `MLFLOW_EXPERIMENT_NAME` | Yes | Pipeline, API | MLflow experiment name (e.g. `customer-churn`) |
| `MODEL_NAME` | Yes | Pipeline | Validated at start-up (the registry name used is `customer_churn_model`) |
| `MLFLOW_TRACKING_USERNAME` | If server needs auth | Pipeline, API | MLflow credentials (DagsHub user) |
| `MLFLOW_TRACKING_PASSWORD` | If server needs auth | Pipeline, API | MLflow credentials (DagsHub token) |
| `PREDICTION_API_URL` | No | Streamlit | Defaults to `http://127.0.0.1:8000/predict` |
| `PREFECT_API_URL` | No | Prefect | e.g. `http://127.0.0.1:4200/api` |

> `.env.example` does not currently list `MLFLOW_EXPERIMENT_NAME` or `PREDICTION_API_URL`; add them to your `.env`.

---


<div align="center">

Built by [rahulvenkatababa-k](https://github.com/rahulvenkatababa-k)

</div>
