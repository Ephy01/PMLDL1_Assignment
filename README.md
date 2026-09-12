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

1. `code/datasets/prepare_data.py`: strip whitespace, remove exact duplicates, replace invalid values. Retain legitimate price extremes. Group identical feature rows into the same 80/20 split to prevent their appearing in both train and test. Save CSVs and `data_report.json`.
  
2. `code/models/train_model.py`: fit imputation and one-hot encoding on training data only. Train Random Forest, compare against a training-median baseline. Print model parameters and test metrics to the execution log (so it can be captured by Airflow). Save metrics to `models/metrics.json`, the complete preprocessing/model pipeline to `models/model.onnx`, and UI metadata to `models/feature_meta.json`.
  
3. `code/deployment/docker-compose.yml`: build FastAPI and Streamlit containers, wait for API health, and serve the model. API validates inputs and returns `price_gbp`.

`notebooks/eda.ipynb` used for data eploration identifying missing values, outliers, distributions.

## Pipeline flow



1. `prepare_data.py`: `load` → `clean` → `split_data` → save CSVs and the report.
2. `train_model.py`: `prepare_features` → `build_pipeline` and `fit` → `export_to_onnx` → `predict_onnx` → `validate_onnx_export` → `evaluate_model` → `build_feature_metadata` → `save_artifacts`.
3. `code/deployment/api/main.py`: request validation, ONNX input construction, prediction.
   
4. `code/deployment/app/app.py`: fetch metadata, show the form, send a request, display the price.
5. `ford_pipeline.py`: schedule the prepare → train → deploy commands in Airflow.



## Run

check requirements.txt

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.12.txt
./run_pipeline.sh
```

- App: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"Focus","year":2018,"transmission":"Manual","mileage":25000,"fuel_type":"Petrol","tax":145,"mpg":57.7,"engine_size":1.0}'
```

## Airflow

```bash
./run_airflow.sh
```

Open http://localhost:8080 and enable `ford_price_pipeline`. The DAG runs prepare → train → deploy every five minutes, with one active run at a time. New installations create it paused. The obsolete `coffee_quality_pipeline` is retired and paused; its historical runs remain in the local Airflow database.

For a complete local DAG run:

```bash
AIRFLOW_HOME="$PWD/services/airflow" .venv/bin/airflow dags test ford_price_pipeline
```

Stop the app and API with `docker compose -f code/deployment/docker-compose.yml down`.

## Validation

```bash
.venv/bin/python code/models/train_model.py
```

Training validates the ONNX graph and compares its predictions with scikit-learn before saving artifacts. It also prints the test metrics and median baseline comparison. The executed EDA notebook checks that identical feature rows do not overlap between train and test.

Seed 42 results: 
MAE £832.70, RMSE £1,203.36, R² 0.9349 vs.  Median baseline: MAE £3,609.51 (76.93% MAE reduction).

## ONNX model

I decided to use ONNX, since it is modern standart.  It acts as a universal translator that lets you train machine learning models in one framework and run them efficiently anywhere - very convient. 


Every training run checks ONNX predictions against scikit-learn on all test rows and three extra cases (missing categories, missing numbers, unseen categories), allowing at most £0.10 absolute difference. The report is saved as `models/onnx_validation.json`; test metrics are computed using ONNX predictions. Small float32 rounding differences are expected. The API response remains `price_gbp`.
