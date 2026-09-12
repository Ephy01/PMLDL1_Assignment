"""Read Ford listings, clean invalid values, and save train/test tables."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ford.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COLUMNS = {
    "model": "model",
    "year": "year",
    "price": "price",
    "transmission": "transmission",
    "mileage": "mileage",
    "fuelType": "fuel_type",
    "tax": "tax",
    "mpg": "mpg",
    "engineSize": "engine_size",
}
CATEGORICAL = ["model", "transmission", "fuel_type"]
NUMERIC = ["year", "mileage", "tax", "mpg", "engine_size"]
TARGET = "price"
COLLECTION_YEAR = 2020


def load(path=RAW_PATH):
    data = pd.read_csv(path)
    data = data[list(COLUMNS)]
    return data.rename(columns=COLUMNS)


def clean(data):
    data = data.copy()

    # Normalize category names and mark empty strings as missing.
    for column in CATEGORICAL:
        data[column] = data[column].str.strip()
        data[column] = data[column].replace("", np.nan)

    # Invalid text and infinite numbers become missing values.
    for column in NUMERIC + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)

    # Training needs a known target. Do not invent prices for these rows.
    valid_price = data[TARGET].notna() & (data[TARGET] > 0)
    data = data[valid_price].copy()

    valid_year = data["year"].between(1900, COLLECTION_YEAR)
    data.loc[~valid_year, "year"] = np.nan

    for column in ["mileage", "tax"]:
        negative_value = data[column] < 0
        data.loc[negative_value, column] = np.nan

    for column in ["mpg", "engine_size"]:
        nonpositive_value = data[column] <= 0
        data.loc[nonpositive_value, column] = np.nan

    # Missing features are filled later using statistics learned on train only.
    data = data.drop_duplicates()
    return data.reset_index(drop=True)


def split_data(data, test_size, seed):
    features = data[CATEGORICAL + NUMERIC]
    groups = pd.util.hash_pandas_object(features, index=False)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_indices, test_indices = next(splitter.split(data, groups=groups))
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    return train_data, test_data


def main(test_size=0.2, seed=42):
    raw_data = load()
    clean_data = clean(raw_data)
    train_data, test_data = split_data(clean_data, test_size, seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_data.to_csv(PROCESSED_DIR / "test.csv", index=False)

    report = {
        "raw_rows": len(raw_data),
        "clean_rows": len(clean_data),
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "missing_after_cleaning": clean_data.isna().sum().to_dict(),
        "split": "grouped by identical feature values",
        "seed": seed,
    }
    report_json = json.dumps(report, indent=2)
    (PROCESSED_DIR / "data_report.json").write_text(report_json)
    print(report_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.test_size, args.seed)
