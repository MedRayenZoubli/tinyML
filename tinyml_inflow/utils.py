"""Small helpers shared by the training, export and test scripts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DATA_DIR, FEATURE_NAMES, SCALER_JSON, SCALER_NPZ


@dataclass(frozen=True)
class Scaler:
    """Min-max scaling constants, fitted on the *training* split only.

    Features are scaled per channel, the target (inflow, m3/s) with one global
    min/max. The same constants are exported to C++ so the ESP32 firmware
    normalises sensor readings identically.
    """

    feature_min: np.ndarray   # shape (4,)
    feature_max: np.ndarray   # shape (4,)
    target_min: float
    target_max: float

    @property
    def feature_span(self) -> np.ndarray:
        return np.maximum(self.feature_max - self.feature_min, 1e-6)

    @property
    def target_span(self) -> float:
        return max(self.target_max - self.target_min, 1e-6)

    def normalize_features(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.feature_min) / self.feature_span).astype(np.float32)

    def normalize_target(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.target_min) / self.target_span).astype(np.float32)

    def denormalize_target(self, y_norm: np.ndarray) -> np.ndarray:
        return (y_norm * self.target_span + self.target_min).astype(np.float32)


def save_scaler(scaler: Scaler) -> None:
    np.savez(
        SCALER_NPZ,
        feature_min=scaler.feature_min.astype(np.float32),
        feature_max=scaler.feature_max.astype(np.float32),
        target_min=np.float32(scaler.target_min),
        target_max=np.float32(scaler.target_max),
    )
    SCALER_JSON.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "feature_min": [float(v) for v in scaler.feature_min],
                "feature_max": [float(v) for v in scaler.feature_max],
                "target_name": "inflow_m3_s",
                "target_min": float(scaler.target_min),
                "target_max": float(scaler.target_max),
            },
            indent=2,
        )
    )


def load_scaler() -> Scaler:
    with np.load(SCALER_NPZ) as z:
        return Scaler(
            feature_min=z["feature_min"].astype(np.float32),
            feature_max=z["feature_max"].astype(np.float32),
            target_min=float(z["target_min"]),
            target_max=float(z["target_max"]),
        )


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the normalised (X, y) arrays of a split: 'train', 'val' or 'test'."""
    x = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")
    return x, y


def make_interpreter(model_path: Path | str | None = None, model_content: bytes | None = None):
    """Create an allocated TFLite interpreter.

    Uses ``tf.lite.Interpreter`` as requested. Recent TensorFlow releases mark it
    deprecated in favour of LiteRT, so fall back to ``ai_edge_litert`` if the
    TensorFlow class is no longer available.
    """
    import tensorflow as tf

    try:
        interpreter_cls = tf.lite.Interpreter
    except AttributeError:  # pragma: no cover - depends on TF version
        from ai_edge_litert.interpreter import Interpreter as interpreter_cls

    if model_content is not None:
        interpreter = interpreter_cls(model_content=model_content)
    else:
        interpreter = interpreter_cls(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter
