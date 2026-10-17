"""HTTP model server: GET /health, GET /spec, POST /predict (window -> plan). Stateless; one request at a time."""

import hmac
import logging
import os
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from ..codec import decode_window, encode_plan
from ..contract import Window
from ..models import Model

logger = logging.getLogger(__name__)


def create_app(model: Model, api_key: str | None = None) -> FastAPI:
    """``api_key`` (env ROLLOUT_API_KEY) makes /spec and /predict require the ``X-Rollout-Key`` header."""
    app = FastAPI(title="SO-101 rollout model server")
    lock = threading.Lock()  # one GPU, one prediction at a time
    spec = model.spec()

    def check_key(request: Request) -> None:
        if api_key and not hmac.compare_digest(request.headers.get("x-rollout-key", ""), api_key):
            raise HTTPException(status_code=401, detail="missing or wrong X-Rollout-Key")

    @app.get("/health")
    def health():
        return {"ok": True, "model": spec.name}

    @app.get("/spec")
    def get_spec(request: Request):
        check_key(request)
        return spec.to_dict()

    @app.post("/predict")
    def predict(request: Request, body: dict):
        check_key(request)
        started = time.perf_counter()
        try:
            window = decode_window(body)
            _validate(window, spec)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=f"invalid window: {error}") from error
        decoded = time.perf_counter()
        with lock:
            plan = model.predict(window)
        plan.timing.update(decode_ms=(decoded - started) * 1e3, server_total_ms=(time.perf_counter() - started) * 1e3)
        return encode_plan(plan)

    return app


def _validate(window: Window, spec) -> None:
    expected = (spec.n_obs_steps, len(spec.action_names))
    if window.states.shape != expected or window.commands.shape != expected:
        raise ValueError(f"states/commands must be {expected}, got {window.states.shape}/{window.commands.shape}")
    if not (np.isfinite(window.states).all() and np.isfinite(window.commands).all()):
        raise ValueError("states/commands must be finite")
    for camera in spec.camera_keys:
        missing = set(spec.snapshot_indices) - set(window.images.get(camera, {}))
        if missing:
            raise ValueError(f"camera {camera!r} lacks images at window positions {sorted(missing)}")


def serve(model: Model, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(model, os.environ.get("ROLLOUT_API_KEY") or None), host=host, port=port, log_level="info")
