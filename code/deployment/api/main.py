

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_PATH = MODEL_DIR / "model.onnx"
METADATA_PATH = MODEL_DIR / "feature_meta.json"

model_session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)
feature_metadata = json.loads(METADATA_PATH.read_text())
app = FastAPI(title="Ford Used Car Price API", version="2.0")


class CarSample(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str = Field(examples=["Focus"])
    year: int = Field(ge=1900, le=2020, examples=[2018])
    transmission: str = Field(examples=["Manual"])
    mileage: int = Field(ge=0, examples=[25000], description="Mileage in miles")
    fuel_type: str = Field(examples=["Petrol"])
    tax: float = Field(ge=0, examples=[145], description="Annual road tax, GBP")
    mpg: float = Field(gt=0, examples=[57.7], description="Miles per UK gallon")
    engine_size: float = Field(
        gt=0,
        examples=[1.0],
        description="Engine displacement, litres",
    )


class Prediction(BaseModel):
    price_gbp: float


def prepare_onnx_inputs(sample: CarSample) -> dict:
    feature_values = sample.model_dump()

    for feature_name, allowed_values in feature_metadata["categorical"].items():
        category = feature_values[feature_name].strip()

        if category not in allowed_values:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported {feature_name}: {category}",
            )

        feature_values[feature_name] = category

    onnx_inputs = {}
    for feature_name in feature_metadata["categorical"]:
        value = feature_values[feature_name]
        onnx_inputs[feature_name] = np.array([[value]], dtype=object)

    for feature_name in feature_metadata["numeric"]:
        value = feature_values[feature_name]
        onnx_inputs[feature_name] = np.array([[value]], dtype=np.float32)

    return onnx_inputs


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "ford-used-car-price",
    }


@app.get("/metadata")
def metadata() -> dict:
    return feature_metadata


@app.post("/predict", response_model=Prediction)
def predict(sample: CarSample) -> Prediction:
    onnx_inputs = prepare_onnx_inputs(sample)
    model_outputs = model_session.run(None, onnx_inputs)
    predicted_price = model_outputs[0].ravel()[0]
    rounded_price = round(float(predicted_price), 2)

    return Prediction(price_gbp=rounded_price)
