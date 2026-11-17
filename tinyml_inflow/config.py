"""Shared configuration, paths and logging for the TinyML inflow-forecasting pipeline.

Every other module imports its constants from here, so the tensor geometry
(24 steps x 4 channels in, 24 steps out) is defined in exactly one place.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

# Silence TensorFlow's C++ info/warning spam. Must be set before TF is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"            # .npy datasets + scaler
ARTIFACT_DIR = PROJECT_DIR / "artifacts"   # models, headers, metrics, plots

# --------------------------------------------------------------------------- #
# Tensor geometry
# --------------------------------------------------------------------------- #
STEP_MINUTES = 5
STEP_SECONDS = STEP_MINUTES * 60
SEQ_LEN = 24        # 24 x 5 min = 2 h look-back (control horizon)
HORIZON = 24        # 24 x 5 min = 2 h forecast
N_FEATURES = 4

FEATURE_NAMES = (
    "precip_rate_mm_hr",
    "basin_volume_m3",
    "discharge_m3_s",
    "soil_saturation_pct",
)

# --------------------------------------------------------------------------- #
# Deployment budget
# --------------------------------------------------------------------------- #
TENSOR_ARENA_BUDGET_BYTES = 32 * 1024

# --------------------------------------------------------------------------- #
# Output files
# --------------------------------------------------------------------------- #
SPLITS = ("train", "val", "test")
SCALER_NPZ = DATA_DIR / "scaler.npz"
SCALER_JSON = DATA_DIR / "scaler.json"
DATASET_META_JSON = DATA_DIR / "dataset_meta.json"

KERAS_MODEL_PATH = ARTIFACT_DIR / "inflow_forecast_model.keras"
TFLITE_MODEL_PATH = ARTIFACT_DIR / "inflow_forecast_model.tflite"
MODEL_HEADER_PATH = ARTIFACT_DIR / "model.h"
SCALER_HEADER_PATH = ARTIFACT_DIR / "scaler_params.h"
TRAIN_HISTORY_JSON = ARTIFACT_DIR / "training_history.json"
TRAIN_CURVES_PNG = ARTIFACT_DIR / "training_curves.png"
EXPORT_INFO_JSON = ARTIFACT_DIR / "export_info.json"
METRICS_JSON = ARTIFACT_DIR / "metrics.json"
COMPARISON_PNG = ARTIFACT_DIR / "inflow_forecast_comparison.png"

DEFAULT_SEED = 42


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a compact timestamped format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy (TensorFlow is seeded separately in train_model)."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
