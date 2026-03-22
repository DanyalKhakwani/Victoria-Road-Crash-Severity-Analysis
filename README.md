# Victorian Road Crash Severity Analysis

Predicting fatal vs non-fatal road crashes using Victorian government crash data.

## Project Structure
```
crash_severity_analysis/
├── src/
│   ├── __init__.py
│   ├── logger.py               ← logging setup
│   ├── exception.py            ← custom exception handling
│   ├── utils.py                ← helper functions (load config etc.)
│   └── components/
│       ├── __init__.py
│       ├── ingest.py           ← connects to SQLite db, runs SQL query, saves raw_data.csv
│       ├── transformation.py   ← cleans, encodes, scales, splits into train/test
│       └── model_trainer.py    ← trains model, evaluates, saves model.pkl
│
├── config/
│   └── config.yaml             ← all experiment settings (model, threshold, drop_columns)
│
├── experiments/                ← JSON logs of every experiment run
│
├── research/
│   └── 01_data_inspection.py   ← EDA and data exploration
│
├── artifacts/                  ← generated at runtime, gitignored
│   ├── raw_data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── main.py                     ← entry point
├── setup.py
├── requirements.txt
└── .gitignore
```

## Data

Victorian government road crash data sourced from VicRoads (2012–2023).
Raw data consists of 5 CSVs merged into a SQLite database with a relational schema.
Database schema and cleaning: [link to other repo]

## Setup
```bash
pip install -r requirements.txt
pip install -e .
```

## Run Pipeline
```bash
py main.py
```

## Current Results

| Model               | Recall (Fatal) | Precision (Fatal) | F1 (Fatal) | ROC AUC |
|---------------------|---------------|-------------------|------------|---------|
| Logistic Regression | 0.77          | 0.05              | 0.09       | 0.84    |

## Experiment Tracking

All model runs are logged to `experiments/` as JSON files including model parameters,
metrics, threshold used, and columns dropped. Change `config/config.yaml` to run
a new experiment without editing any Python files.