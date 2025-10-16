import io
import json
import os
import threading
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic import field_validator
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B4_Weights


# CONFIG & DEVICE

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

MODEL_VARIANTS: Dict[str, Dict[str, object]] = {
    "b0": {
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "builder": models.efficientnet_b0,
        "title_suffix": "EfficientNet-B0",
    },
    "b4": {
        "weights": EfficientNet_B4_Weights.IMAGENET1K_V1,
        "builder": models.efficientnet_b4,
        "title_suffix": "EfficientNet-B4",
    },
}

CONFIG_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[nn.Module, transforms.Compose]] = {}


def _load_config_or_defaults(path: str) -> Dict[str, object]:
    defaults = {"model_variant": "b4", "threshold": 0.5}
    if not os.path.exists(path):
        print(f"[config] No config file at {path}. Using defaults: {defaults}")
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
            if not isinstance(cfg, dict):
                raise ValueError("Top-level JSON must be an object.")
            for k, v in defaults.items():
                cfg.setdefault(k, v)
            return cfg
    except Exception as exc:
        print(f"[config] Invalid config at {path}: {exc}. Using defaults: {defaults}")
        return defaults


def normalize_variant(value: object) -> str:
    variant = str(value).lower()
    if variant not in MODEL_VARIANTS:
        supported = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unsupported model_variant '{variant}'. Supported: {supported}")
    return variant


def resolve_checkpoint_path(config: Dict[str, object], variant: str) -> str:
    path = config.get("checkpoint_path")
    return str(path) if path is not None else f"model_{variant}.pth"


def compose_app_title(variant: str) -> str:
    return f"Denum-prefilter"


def build_model_and_transforms(variant: str) -> Tuple[nn.Module, transforms.Compose]:
    variant_data = MODEL_VARIANTS[variant]
    weights = variant_data["weights"]

    t = weights.transforms()
    image_size = t.crop_size[0]
    mean = t.mean
    std = t.std

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    builder = variant_data["builder"]
    model = builder(weights=None)

    last_linear = model.classifier[-1]
    in_feats = getattr(last_linear, "in_features")
    model.classifier[-1] = nn.Linear(in_feats, 1)

    return model, eval_tf


def _load_state_dict(checkpoint_path: str):
    try:
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def load_model_bundle(variant: str, checkpoint_path: str) -> Tuple[nn.Module, transforms.Compose]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model, eval_tf = build_model_and_transforms(variant)
    state = _load_state_dict(checkpoint_path)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("load_state_dict mismatches:",
              f"\n  missing:   {missing}",
              f"\n  unexpected:{unexpected}")

    model.to(DEVICE).eval()
    return model, eval_tf


def get_or_load_bundle(variant: str, checkpoint_path: str) -> Tuple[nn.Module, transforms.Compose]:
    key = (variant, checkpoint_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model, eval_tf = load_model_bundle(variant, checkpoint_path)
    _MODEL_CACHE[key] = (model, eval_tf)
    return model, eval_tf

# INITIAL RUNTIME CONFIG & DEFAULT MODEL

CONFIG = _load_config_or_defaults(CONFIG_PATH)
MODEL_VARIANT = normalize_variant(CONFIG.get("model_variant", "b4"))
THRESHOLD = float(CONFIG.get("threshold", 0.5))
CHECKPOINT_PATH = resolve_checkpoint_path(CONFIG, MODEL_VARIANT)

try:
    _model, EVAL_TRANSFORMS = get_or_load_bundle(MODEL_VARIANT, CHECKPOINT_PATH)
except FileNotFoundError:
    print(f"[startup] No checkpoint found at {CHECKPOINT_PATH}. "
          f"Server will still start; first request with a valid checkpoint will load on demand.")
    _model, EVAL_TRANSFORMS = build_model_and_transforms(MODEL_VARIANT)

# SCHEMAS

class ConfigResponse(BaseModel):
    model_variant: str
    threshold: float
    checkpoint_path: str


class ConfigUpdateRequest(BaseModel):
    model_variant: Optional[str] = None
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    checkpoint_path: Optional[str] = None

    @field_validator("model_variant")
    @classmethod
    def _validate_variant(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return normalize_variant(value)

    @field_validator("checkpoint_path")
    @classmethod
    def _validate_checkpoint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        path = value.strip()
        if not path:
            raise ValueError("checkpoint_path cannot be empty")
        return path


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_variant: str
    threshold: float

# APP

app = FastAPI(title=compose_app_title(MODEL_VARIANT), version="1.0")


def current_config_response() -> ConfigResponse:
    return ConfigResponse(
        model_variant=MODEL_VARIANT,
        threshold=THRESHOLD,
        checkpoint_path=CHECKPOINT_PATH,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/config", response_model=ConfigResponse)
async def get_runtime_config() -> ConfigResponse:
    return current_config_response()


@app.post("/config", response_model=ConfigResponse)
async def update_runtime_config(update: ConfigUpdateRequest) -> ConfigResponse:
    global MODEL_VARIANT, THRESHOLD, CHECKPOINT_PATH, _model, EVAL_TRANSFORMS, CONFIG

    with CONFIG_LOCK:
        new_variant = MODEL_VARIANT if update.model_variant is None else update.model_variant
        new_threshold = THRESHOLD if update.threshold is None else float(update.threshold)

        config_snapshot = dict(CONFIG)
        config_snapshot["model_variant"] = new_variant
        config_snapshot["threshold"] = new_threshold

        if "checkpoint_path" in update.__fields_set__:
            if update.checkpoint_path is None:
                config_snapshot.pop("checkpoint_path", None)
            else:
                config_snapshot["checkpoint_path"] = update.checkpoint_path

        resolved_checkpoint = resolve_checkpoint_path(config_snapshot, new_variant)
        reload_needed = (new_variant != MODEL_VARIANT) or (resolved_checkpoint != CHECKPOINT_PATH)

        new_model: Optional[nn.Module] = None
        new_transforms: Optional[transforms.Compose] = None

        if reload_needed:
            try:
                new_model, new_transforms = get_or_load_bundle(new_variant, resolved_checkpoint)
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Persist config
        CONFIG.clear()
        CONFIG.update(config_snapshot)

        # Swap globals
        if reload_needed and new_model is not None and new_transforms is not None:
            _model = new_model
            EVAL_TRANSFORMS = new_transforms

        THRESHOLD = new_threshold
        MODEL_VARIANT = new_variant
        CHECKPOINT_PATH = resolved_checkpoint
        app.title = compose_app_title(MODEL_VARIANT)

    return current_config_response()

@app.post("/predict", response_model=PredictionResponse)
async def predict_notation(
    file: UploadFile = File(...),
    model_variant: Optional[str] = Form(None, example=None, description='Optional. "b0" or "b4".'),
    threshold: Optional[str]     = Form(None, example=None, description="Optional. Float in [0,1]."),
    checkpoint_path: Optional[str] = Form(None, example=None, description="Optional. Path to .pth."),
    persist: bool = Form(False, description="If true, make these overrides the new defaults"),
):
    global MODEL_VARIANT, THRESHOLD, CHECKPOINT_PATH, _model, EVAL_TRANSFORMS, CONFIG

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image.")

    def _clean(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s2 = s.strip()
        return None if s2 == "" else s2

    model_variant = _clean(model_variant)
    threshold_str = _clean(threshold)
    checkpoint_path = _clean(checkpoint_path)

    try:
        req_variant = normalize_variant(model_variant) if model_variant is not None else MODEL_VARIANT
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    temp_config = dict(CONFIG)
    temp_config["model_variant"] = req_variant
    if checkpoint_path is not None:
        temp_config["checkpoint_path"] = checkpoint_path
    resolved_ckpt = resolve_checkpoint_path(temp_config, req_variant)

    if threshold_str is None:
        req_threshold = THRESHOLD
    else:
        try:
            req_threshold = float(threshold_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="threshold must be a float (e.g., 0.5)")
        if not (0.0 <= req_threshold <= 1.0):
            raise HTTPException(status_code=400, detail="threshold must be in [0,1]")

    try:
        model, eval_tf = get_or_load_bundle(req_variant, resolved_ckpt)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tensor = eval_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        probability = torch.sigmoid(logit).item()
    prediction_label = "YES" if probability >= req_threshold else "NO"

    if persist:
        with CONFIG_LOCK:
            CONFIG["model_variant"] = req_variant
            CONFIG["threshold"] = req_threshold
            if checkpoint_path is not None:
                CONFIG["checkpoint_path"] = checkpoint_path
            MODEL_VARIANT = req_variant
            THRESHOLD = req_threshold
            CHECKPOINT_PATH = resolved_ckpt
            _model, EVAL_TRANSFORMS = model, eval_tf
            app.title = compose_app_title(MODEL_VARIANT)

    return {
        "prediction": prediction_label,
        "probability": probability,
        "model_variant": req_variant,
        "threshold": req_threshold
    }


# MAIN

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
