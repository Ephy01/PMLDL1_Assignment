# PMLDL Assignment 1 — Ford Used Car Price Pipeline

Predict the advertised price in GBP of a used Ford from eight vehicle attributes.

Dataset: [100,000 UK Used Car Dataset — Aditya, Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). I used only `ford.csv`: 17,965 rows, 9 columns. This is a historical 2020 snapshot: predictions are neither current valuations nor completed-sale prices.

## Features and target

| Field | Meaning |
|---|---|
| model | Ford model |
| year | Vehicle year |
| transmission | Manual, Automatic, Semi-Auto |
| mileage | Miles driven |
| fuel_type | Fuel category |
| tax | Annual road tax, GBP |
| mpg | Miles per UK gallon |
| engine_size | Engine displacement, litres |
| price | Target: advertised price, GBP |

## Pipeline

1. `code/datasets/prepare_data.py`: strip whitespace, remove exact duplicates and rows with missing or nonpositive prices, and mark invalid feature values as missing. Retain legitimate price extremes. Group identical feature rows into the same approximately 80/20 split to prevent their appearing in both train and test. Save CSVs and `data_report.json`. Missing feature values are filled during Stage 2 using statistics learned on training data only.

2. `code/models/train_model.py`: fit imputation and one-hot encoding on training data only. Train Random Forest, compare against a training-median baseline. Print model parameters and test metrics to the execution log (so it can be captured by Airflow). Save metrics to `models/metrics.json`, the complete preprocessing/model pipeline to `models/model.onnx`, and UI metadata to `models/feature_meta.json`.

3. `code/deployment/docker-compose.yml`: build FastAPI and Streamlit containers, wait for API health, and serve the model. API validates inputs and returns `price_gbp`.

`notebooks/eda.ipynb` explores missing values, outliers, and feature distributions.

## Pipeline flow

1. `prepare_data.py`: `load` → `clean` → `split_data` → save CSVs and the report.
2. `train_model.py`: `prepare_features` → `build_pipeline` and `fit` → `export_to_onnx` → `predict_onnx` → `validate_onnx_export` → `evaluate_model` → `build_feature_metadata` → `save_artifacts`.
3. `code/deployment/api/main.py`: request validation, ONNX input construction, prediction.
4. `code/deployment/app/app.py`: fetch metadata, show the form, send a request, display the price.
5. `ford_pipeline.py`: schedule the prepare → train → deploy commands in Airflow.

## Run

Use Python 3.12 and a running Docker installation with Docker Compose v2. Airflow runs locally in the virtual environment; FastAPI and Streamlit run in separate Docker containers. On Windows, use WSL2.

Clone the repository into a path, run the following commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "apache-airflow==3.3.1" "apache-airflow-providers-standard==1.17.0" \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.12.txt
python -m pip install -r requirements.txt
python -m pip check
bash run_pipeline.sh
```

Install Airflow with its constraints first, then install the project dependencies without that constraint file. The project pins newer NumPy and scikit-learn versions than the Airflow constraints, so applying those constraints to the entire `requirements.txt` causes a dependency conflict. Check: [Airflow installation guidance](https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.12.txt).

`run_pipeline.sh` prepares the data, trains and exports the model, then builds and starts both containers. Each run overwrites the processed CSVs and model artifacts.

- App: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"Focus","year":2018,"transmission":"Manual","mileage":25000,"fuel_type":"Petrol","tax":145,"mpg":57.7,"engine_size":1.0}'
```

## Airflow

After installing the dependencies above, run this from the repository root:

```bash
bash run_airflow.sh
```

The first start initializes the local Airflow database and starts the scheduler and web interface. Open http://localhost:8080, log in as `admin` with the password printed in the terminal on first creation, and enable `ford_price_pipeline`. On later starts, look up the saved password in `services/airflow/simple_auth_manager_passwords.json.generated`; this file is excluded from Git.

The DAG is initially paused. Once enabled, it runs prepare → train → deploy every five minutes, with one active run at a time. Keep Airflow and Docker running for the complete automated pipeline. Every scheduled run retrains the model and deploys the resulting artifacts.

After Airflow has been initialized, you can also run one complete DAG manually:

```bash
AIRFLOW_HOME="$PWD/services/airflow" .venv/bin/airflow dags test ford_price_pipeline
```

This command also prepares data, retrains the model, and deploys the containers.

## Stop

Stop Airflow with `Ctrl+C` in the terminal running `run_airflow.sh`, then stop the app and API:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

If you keep Airflow running, pause `ford_price_pipeline` and wait for any active run to finish before stopping the containers; otherwise, a deployment task can start them again.

## Validation

Seed 42 results:
MAE £832.70, RMSE £1,203.36, R² 0.9349. Median baseline: MAE £3,609.51 (76.93% MAE reduction).

## ONNX model

I use ONNX to package the complete preprocessing and regression pipeline in one file. The API loads it with ONNX Runtime and does not need scikit-learn installed.

Every training run checks ONNX predictions against scikit-learn on all test rows and three extra cases (missing categories, missing numbers, unseen categories), allowing at most £0.10 absolute difference. The report is saved as `models/onnx_validation.json`; test metrics are computed using ONNX predictions. Small float32 rounding differences are expected. The API response remains `price_gbp`.
