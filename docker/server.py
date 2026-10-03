"""FastAPI server exposing the NS-VLA JAX model."""

from __future__ import annotations

import base64
import binascii
import os
from typing import List, Optional

import jax
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docker.model import (
    ACTION_DIM,
    PRIMITIVES,
    NSVLAJaxModel,
    load_image_from_bytes,
)

app = FastAPI(
    title="NS-VLA JAX Inference API",
    description="HTTP wrapper around the NS-VLA Vision-Language-Action model.",
    version="0.1.0",
)

HORIZON = int(os.environ.get("NSVLA_HORIZON", "8"))
_model = NSVLAJaxModel(horizon=HORIZON)


class PredictJsonRequest(BaseModel):
    instruction: str = Field(..., description="Natural-language task instruction.")
    image_base64: str = Field(..., description="Base64-encoded RGB image.")


class PredictResponse(BaseModel):
    primitive: str
    primitive_scores: dict
    actions: List[List[float]]
    horizon: int
    action_dim: int
    latency_ms: float


class InfoResponse(BaseModel):
    name: str
    version: str
    backend: str
    jax_devices: List[str]
    horizon: int
    action_dim: int
    primitives: List[str]
    weights_loaded: bool
    notes: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        name="NS-VLA",
        version="0.1.0",
        backend="jax",
        jax_devices=[str(d) for d in jax.devices()],
        horizon=HORIZON,
        action_dim=ACTION_DIM,
        primitives=PRIMITIVES,
        weights_loaded=False,
        notes=(
            "Running the deterministic JAX placeholder. The published NS-VLA "
            "weights are not yet released; swap docker/model.py when they are."
        ),
    )


@app.get("/primitives")
def primitives():
    return {"primitives": PRIMITIVES}


def _run_prediction(image_bytes: bytes, instruction: str) -> PredictResponse:
    try:
        image = load_image_from_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    result = _model.predict(image, instruction)
    scores = {p: float(s) for p, s in zip(PRIMITIVES, result.primitive_logits)}
    return PredictResponse(
        primitive=result.primitive,
        primitive_scores=scores,
        actions=result.actions.tolist(),
        horizon=HORIZON,
        action_dim=ACTION_DIM,
        latency_ms=result.latency_ms,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_json(req: PredictJsonRequest):
    try:
        image_bytes = base64.b64decode(req.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")
    return _run_prediction(image_bytes, req.instruction)


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    instruction: str = Form(...),
    image: UploadFile = File(...),
):
    data = await image.read()
    return _run_prediction(data, instruction)
