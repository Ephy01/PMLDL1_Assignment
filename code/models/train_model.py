"""Train a Ford price model, check its ONNX export, and save the results."""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "datasets"))
from prepare_data import CATEGORICAL, NUMERIC, TARGET

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 18,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}
ONNX_OPSET = 17
ONNX_TOLERANCE_GBP = 0.1

#prepare data to be used by ONNX runtime
def prepare_features(data):
    features = data[CATEGORICAL + NUMERIC].copy()
    for column in CATEGORICAL:
        features[column] = features[column].fillna("").astype(str)
    for column in NUMERIC:
        features[column] = features[column].astype(np.float32)

    return features


def build_pipeline():
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values="")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_imputer = SimpleImputer(strategy="median")

    preprocessing = ColumnTransformer([
        ("cat", categorical_steps, CATEGORICAL),
        ("num", numeric_imputer, NUMERIC),
    ])
    forest = RandomForestRegressor(**MODEL_PARAMS)

    return Pipeline([
        ("features", preprocessing),
        ("model", forest),
    ])


def export_to_onnx(pipeline):
    input_types = []
    for column in CATEGORICAL:
        input_types.append((column, StringTensorType([None, 1])))
    for column in NUMERIC:
        input_types.append((column, FloatTensorType([None, 1])))

    graph = convert_sklearn(
        pipeline,
        initial_types=input_types,
        target_opset=ONNX_OPSET,
    )
    onnx.checker.check_model(graph)
    return graph.SerializeToString()


def predict_onnx(session, features):
    inputs = {}
    for column in CATEGORICAL:
        values = features[column].to_numpy(dtype=object)
        inputs[column] = values.reshape(-1, 1)
    for column in NUMERIC:
        values = features[column].to_numpy(dtype=np.float32)
        inputs[column] = values.reshape(-1, 1)

    outputs = session.run(None, inputs)
    predictions = outputs[0].ravel()
    return predictions


def validate_onnx_export(pipeline, session, test_features, onnx_predictions):
    sklearn_predictions = pipeline.predict(test_features)
    np.testing.assert_allclose(
        onnx_predictions,
        sklearn_predictions,
        rtol=0,
        atol=ONNX_TOLERANCE_GBP,
    )
    edge_cases = test_features.head(3).copy()
    edge_cases.loc[edge_cases.index[0], CATEGORICAL] = ""
    edge_cases.loc[edge_cases.index[1], NUMERIC] = np.nan
    edge_cases.loc[edge_cases.index[2], CATEGORICAL] = "unseen-category"

    onnx_edge_predictions = predict_onnx(session, edge_cases)
    sklearn_edge_predictions = pipeline.predict(edge_cases)
    np.testing.assert_allclose(
        onnx_edge_predictions,
        sklearn_edge_predictions,
        rtol=0,
        atol=ONNX_TOLERANCE_GBP,
    )

    test_difference = np.max(np.abs(onnx_predictions - sklearn_predictions))
    edge_difference = np.max(np.abs(onnx_edge_predictions - sklearn_edge_predictions))
    max_difference = float(max(test_difference, edge_difference))

    return {
        "test_rows": len(test_features),
        "edge_case_rows": len(edge_cases),
        "max_absolute_difference_gbp": max_difference,
        "tolerance_gbp": ONNX_TOLERANCE_GBP,
        "opset": ONNX_OPSET,
    }


def evaluate_model(train_features, train_prices, test_features, test_prices, predictions):
    baseline = DummyRegressor(strategy="median")
    baseline.fit(train_features, train_prices)
    baseline_predictions = baseline.predict(test_features)
    baseline_mae = float(mean_absolute_error(test_prices, baseline_predictions))

    mae = float(mean_absolute_error(test_prices, predictions))
    rmse = float(np.sqrt(mean_squared_error(test_prices, predictions)))
    r2 = float(r2_score(test_prices, predictions))
    improvement = 100 * (1 - mae / baseline_mae)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "baseline_mae": baseline_mae,
        "mae_improvement_pct": improvement,
    }


def build_feature_metadata(train_data, metrics):
    categories = {}
    for column in CATEGORICAL:
        values = train_data[column].dropna().unique().tolist()
        categories[column] = sorted(values)

    numeric_ranges = {}
    for column in NUMERIC:
        values = train_data[column]
        numeric_ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
        }

    return {
        "categorical": categories,
        "numeric": numeric_ranges,
        "format": "onnx",
        "target": TARGET,
        "currency": "GBP",
        "dataset": "Kaggle 100,000 UK Used Car Dataset / ford.csv",
        "collection_year": 2020,
        "metrics": metrics,
    }


def save_artifacts(model_bytes, metrics, feature_metadata, export_report):
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "model.onnx").write_bytes(model_bytes)

    reports = {
        "metrics.json": metrics,
        "feature_meta.json": feature_metadata,
        "onnx_validation.json": export_report,
    }
    for filename, report in reports.items():
        path = MODELS_DIR / filename
        path.write_text(json.dumps(report, indent=2))


def main():
    train_data = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_data = pd.read_csv(PROCESSED_DIR / "test.csv")
    train_features = prepare_features(train_data)
    test_features = prepare_features(test_data)
    train_prices = train_data[TARGET]
    test_prices = test_data[TARGET]

    print("model params:", json.dumps(MODEL_PARAMS, indent=2))
    pipeline = build_pipeline()
    pipeline.fit(train_features, train_prices)

    model_bytes = export_to_onnx(pipeline)
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    predictions = predict_onnx(session, test_features)
    export_report = validate_onnx_export(pipeline, session, test_features, predictions)

    metrics = evaluate_model(
        train_features, train_prices, test_features, test_prices, predictions,
    )
    print("test metrics:", json.dumps(metrics, indent=2))
    feature_metadata = build_feature_metadata(train_data, metrics)
    save_artifacts(model_bytes, metrics, feature_metadata, export_report)


if __name__ == "__main__":
    main()
