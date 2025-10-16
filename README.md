# Denum-prefilter API

This repository hosts a FastAPI application that exposes convolutional neural networks trained to prefilter document images by detecting the presence of musical notation. The service is designed as a high-recall filter that flags pages likely containing notation while maintaining a useful true-negative rate, allowing downstream pipelines to concentrate on genuinely musical content.

Model architecture, checkpoint, and decision threshold are configured via `config.json`. At runtime the service loads a binary classifier fine-tuned on proprietary datasets of scanned pages with and without notation and predicts whether musical notation is present (`YES`) or absent (`NO`) given a single RGB image.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model checkpoints (`model_b0.pth`, `model_b4.pth`) must be present in the working directory (or another location referenced in `config.json`). If a referenced checkpoint is missing at startup the server still boots, and the first request that points at a valid file will trigger the load.

## Configuration

All runtime settings live in `config.json`:

```json
{
  "model_variant": "b4",
  "threshold": 0.5
}
```

- `model_variant`: `"b0"` or `"b4"` to select the EfficientNet backbone.
- `threshold`: sigmoid cutoff used to map probabilities to `YES`/`NO`.
- `checkpoint_path` (optional): filesystem path to the `.pth` checkpoint to load if you don't want the default `model_<variant>.pth`.

The file location can be overridden by setting the `CONFIG_PATH` environment variable before launching the app.

## API Endpoints

- `GET /health` returns a simple status payload together with the device currently selected (`mps`, `cuda`, or `cpu`).
- `GET /config` returns the active `model_variant`, `threshold`, and resolved `checkpoint_path`.
- `POST /config` accepts any subset of those fields to update the runtime configuration. Leave a field out to keep the prior value or send `"checkpoint_path": null` to clear a previously set override.

Example switch to the B0 model with a lower threshold:

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"model_variant": "b0", "threshold": 0.4}'
```

A checkpoint path override can be supplied in the same payload if needed.

## Running the APIs

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The device is chosen automatically in priority order `Apple MPS → CUDA → CPU`.

## Inference Contract

- **Endpoint:** `POST /predict`
- **Payload:** multipart form with an `image/*` file part named `file` and optional form fields:
  - `model_variant`: overrides the configured variant (`"b0"` or `"b4"`).
  - `threshold`: overrides the sigmoid cutoff; must parse as a float in `[0, 1]`.
  - `checkpoint_path`: points to an alternate checkpoint file.
  - `persist`: set to `"true"` to make the supplied overrides the new defaults for subsequent requests and for `GET /config`.
- **Response:**

```json
{
  "prediction": "YES",
  "probability": 0.87,
  "model_variant": "b4",
  "threshold": 0.5
}
```

`prediction` reports the thresholded decision and `probability` reports the sigmoid score ∈ [0, 1]. The response also echoes the effective `model_variant` and `threshold` that produced the decision so you can confirm any overrides.

## Evaluation Summary

Performance metrics were computed on two complementary datasets:

1. **Balanced validation suite** with a 50/50 split between notation and non-notation pages.
2. **Edge-case stress test** of 100k pages that earlier model generations all classified as positives. Only 1.5% of this corpus contains actual notation.

### Balanced validation (50/50)

| Model | Threshold | Throughput (img/s) | Recall | TNR |
|:------|:----------|:-------------------|:-------|:----|
| EfficientNet-B0 | 0.35 | ~36 | ≥ 0.98 | ~0.52 |
| EfficientNet-B4 | 0.60 | ~20 | ~0.99 | ~0.78 |

### Edge-case corpus (1.5% positives)

| Model | Threshold | Recall | TNR |
|:------|:----------|:-------|:----|
| EfficientNet-B0 | 0.50 | ~0.97 | ~0.36 |
| EfficientNet-B4 | 0.50 | ~0.98 | ~0.79 |

The edge-case dataset is intentionally adversarial: every sample was previously flagged as positive by legacy models, making the operating point of this generation particularly salient. Both current models retain high recall while recovering substantial specificity, with the EfficientNet-B4 variant nearly recovering four-fifths of the false positives.

## Design Notes

- **Objective:** Maximize recall under tight latency constraints while improving TNR on difficult negatives to reduce downstream review costs.
- **Preprocessing:** Inputs are resized to a square resolution matching the model’s crop size (224 for B0, 380 for B4) using bicubic interpolation with antialiasing and normalized with ImageNet statistics. This data-driven variant outperformed canonical EfficientNet preprocessing on our datasets, improving recall and TNR on hard negatives.
- **Calibration:** Thresholds are configurable per deployment through `config.json`; values above reflect empirically optimized operating points for the reported datasets.
- **Model selection:** EfficientNet-B0 provides a rapid screening layer suitable for high-throughput scraping pipelines, while EfficientNet-B4 acts as a confirmatory stage where additional latency is acceptable in exchange for sharper discrimination. Switch between them by updating `model_variant` in the configuration.

## Limitations and Future Work

- The reported metrics derive from proprietary datasets; broader generalization depends on the similarity between deployment data and the curated corpora.
- The service currently handles single images per request; batching and async streaming could further amplify throughput.
