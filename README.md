# IMPORTANT: Local storage requirement

**You need at least 400 GB of free disk space** to work with this project locally (dataset archives + extracted images + DVC cache + artifacts/checkpoints).

---

# ECG Classifier

This repository contains an end-to-end MLOps-ready training pipeline for **ECG image classification** with a future goal of integrating the model into a **Telegram bot**. The system takes an **ECG image** as input and outputs **probabilities over 5 diagnostic classes**:

- **4 pathology classes**
- **1 normal class**

The project is educational, but the intended user perspective is a patient-facing interface (Telegram bot).

---

## Project specification

### Problem statement

This project solve a **5-class image classification** problem: given an ECG image uploaded by a user, predict which diagnostic superclass it belongs to:

- `CD`
- `HYP`
- `MI`
- `STTC`
- `NORM`

The output of the system is a probability distribution over the 5 classes, plus the predicted class (argmax).

### Input and output formats

**Input**
- A single ECG image sent to the inference service (future Telegram bot integration).
- Supported formats: `PNG`, `JPG`, `JPEG`
- Images are preprocessed to a fixed size (default: **224×224**) and normalized.

**Output**
- A probability vector of length 5 (one probability per class).
- Example JSON-like structure:
  ```json
  {
    "probabilities": {
      "CD": 0.12,
      "HYP": 0.05,
      "MI": 0.63,
      "NORM": 0.10,
      "STTC": 0.10
    },
    "predicted_class": "MI"
  }
  ```

### Metrics
Because this is a medical classification problem, our primary focus is to reduce false negatives for pathological cases. Therefore:
- Primary metric: `Recall` (macro)
- Additional metrics: `Precision` (macro), `ROC-AUC` (macro)
All metrics are computed and logged during training/validation/testing via **PyTorch Lightning** + **MLflow**.

### Validation strategy:
The dataset is split into:
- `train`: 0.8
- `val`: 0.1
- `test`: 0.1
In that project used a stratified split per class (same class distribution across splits). The split is reproducible via a fixed random seed (default: `seed=42`), and the resulting CSV split files are stored under:
- `artifacts/splits/<split_name>/train.csv`
- `artifacts/splits/<split_name>/val.csv`
- `artifacts/splits/<split_name>/test.csv`

### Datasets
The original signal source is PTB-XL:
- https://physionet.org/content/ptb-xl/1.0.3/

For this repository was used pre-generated ECG images, already organized by class folders:
ecg_img/
  CD/
  HYP/
  MI/
  NORM/
  STTC/
The dataset is tracked with DVC, and stored in an S3-compatible backend (MinIO).

You can see zip by the link:
- https://disk.360.yandex.ru/d/S99scWCwHLGEfw

### Modeling
#### Baseline
A simple CNN (`SimpleCnn`) is used as a baseline:
- 3 convolutional blocks (Conv → BN → ReLU → MaxPool)
- global pooling + 2-layer classifier head
- trained with Cross Entropy loss

#### Main model
A Vision Transformer (ViT) model from `timm`:
- default: `vit_base_patch16_224`
- pretrained initialization supported
- trained via PyTorch Lightning

Both models are trained using the same training/evaluation pipeline.

### Deployment / usage format

The training and evaluation workflows are implemented as CLI commands via:
- `Hydra` configuration system
- `Fire` for command dispatch
- A single entry point: `ecg_classifier/commands.py`

ML experiment tracking is done with MLflow (tracking server expected at http://127.0.0.1:8080 by default).

---

## Technical guide

### Repository structure

Key folders:
- `ecg_classifier/` — source code (Hydra configs, training pipeline, models)
- `ecg_img/` — ECG images dataset (tracked with DVC)
- `artifacts/` — splits, checkpoints, metrics, run outputs (should be tracked with DVC)
- `infra/` — docker-compose for MinIO + MLflow

### Setup

**1) Prerequisites**

- Python 3.11
- Poetry
- Git
- Docker + docker-compose (recommended, for MinIO + MLflow)
- (Optional) NVIDIA drivers + CUDA for GPU training

**2) Install dependencies**

From the repo root:
```bash
poetry install
```
Choose one of the torch installation modes:
**CPU-only**
```bash
poetry install --with cpu
```
**GPU (CUDA 12.1 wheels)**
```bash
poetry install --with gpu
```
**3) Pre-commit hooks**

Install git hooks:
```bash
poetry run pre-commit install
```
Run on all files:
```bash
poetry run pre-commit run --all-files
```
This project uses:
- `ruff` (lint + format) for Python
- `prettier` for JSON/YAML/Markdown
- standard `pre-commit-hooks`

**4) Start MinIO + MLflow (recommended)**
This project assume that reviewers already have MinIO and MLflow running. For local development, you can use:
```bash
docker compose -f infra/docker-compose.yml up -d
```

Services:
- MinIO API: `http://127.0.0.1:9000`
- MinIO Console: `http://127.0.0.1:9001`
- MLflow: `http://127.0.0.1:8080`
The compose file also creates buckets:
- `dvc`
- `mlflow`

**5) Configure DVC remote (MinIO)**
Initialize DVC once:
```bash
dvc init
```
Add MinIO remote:
```bash
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://127.0.0.1:9000
dvc remote modify minio access_key_id minio
dvc remote modify minio secret_access_key minio123
```

**6) Pull dataset from DVC**
If `ecg_img` is tracked by DVC, restore it:
```bash
dvc pull ecg_img.dvc
```
If DVC pull fails or dataset is missing, the pipeline can fall back to downloading a multipart archive from Yandex.Disk (see download_data command). You need to set download.public_url.

### Train

All commands are executed via a single entry point:
```bash
poetry run python -m ecg_classifier.commands <command> [hydra_overrides...]
```
Hydra overrides are appended as `key=value`, e.g. `model=vit train.max_epochs=20`.

**1) (Optional) Download dataset from Yandex.Disk**

This is a fallback method when `dvc pull` is not available.
```bash
poetry run python -m ecg_classifier.commands download_data download.public_url="https://disk.360.yandex.ru/d/S99scWCwHLGEfw"
```
Notes:
- The download is multipart (`.zip`, `.z01`, `.z02`, `.z03`)
- Extraction uses 7-Zip; configure:
    - `download.seven_zip_path` (Windows path) or ensure 7z is in PATH
 
**2) Create train/val/test split**

By default, split ratios are `0.8/0.1/0.1` and split name is `split_v1`.
```bash
poetry run python -m ecg_classifier.commands split
```
The split CSVs are written into:
- `artifacts/splits/split_v1/train.csv`
- `artifacts/splits/split_v1/val.csv`
- `artifacts/splits/split_v1/test.csv`

You can change split parameters with Hydra:
```bash
poetry run python -m ecg_classifier.commands split split.output_name=split_v2 split.train_ratio=0.75 split.val_ratio=0.15 split.test_ratio=0.10
```

**3) Train the baseline CNN**

```bash
poetry run python -m ecg_classifier.commands train model=cnn
```
Key defaults are in:
- `ecg_classifier/conf/model/cnn.yaml`
- `ecg_classifier/conf/train/default.yaml`
Outputs:
- Best checkpoint saved under `artifacts/checkpoints/cnn/<date>/cnn.ckpt` (exact path printed)

**4) Train the ViT model**

```bash
poetry run python -m ecg_classifier.commands train model=vit
```
Defaults are in:
- `ecg_classifier/conf/model/vit.yaml`

Recommended overrides (if you hit VRAM limits):
```bash
poetry run python -m ecg_classifier.commands train model=vit model.batch_size=8 train.accumulate_grad_batches=2
```

**5) Evaluate a trained model on test set**

Pass the checkpoint path printed during training:
```bash
poetry run python -m ecg_classifier.commands evaluate --checkpoint_path="artifacts/checkpoints/cnn/<date>/cnn.ckpt" model=cnn
```
The evaluation metrics are saved to:
- `artifacts/metrics/<model_name>/test_metrics.json`

**6) MLflow tracking**

By default, MLflow is expected at:
- `http://127.0.0.1:8080`
Config: `ecg_classifier/conf/mlflow/default.yaml`

During training:
- metrics (`loss`, `recall`, `precision`, `roc-auc`) are logged per stage
- hyperparameters are logged
- `git_commit_id` is saved as an MLflow run tag

### Data & artifacts versioning (DVC)
In this project track:
- dataset (`ecg_img/`)
- splits (`artifacts/splits/`)
- checkpoints (`artifacts/checkpoints/`)
- metrics (`artifacts/metrics/`)
- (optional) downloaded archives (`artifacts/downloads/`)

Example workflow:
```bash
dvc add artifacts/splits artifacts/checkpoints artifacts/metrics
git add artifacts/*.dvc .gitignore
git commit -m "Track training artifacts with DVC"
dvc push
```
---

## Hardware notes
This project is tested on:
- CPU: Intel i9-12900HK
- GPU: RTX 3080 Ti Laptop
- RAM: 64 GB
