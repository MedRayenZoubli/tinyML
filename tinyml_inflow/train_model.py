"""Train the lightweight 1D-CNN inflow forecaster (pipeline step 2).

Architecture (fixed by the 32 KB tensor-arena budget, ~4.5 k parameters):

    Input (24, 4)
    Conv1D(16, k=3, same, relu) -> MaxPooling1D(2) -> Conv1D(8, k=3, same, relu)
    -> Flatten -> Dense(32, relu) -> Dense(24)   # linear regression head

Adam + MSE, early stopping on the validation split (restoring the best weights).
Outputs (artifacts/): inflow_forecast_model.keras, training_history.json, training_curves.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    DEFAULT_SEED,
    HORIZON,
    KERAS_MODEL_PATH,
    N_FEATURES,
    SEQ_LEN,
    TRAIN_CURVES_PNG,
    TRAIN_HISTORY_JSON,
    ensure_dirs,
    get_logger,
    seed_everything,
)
from plot_style import BLUE, INK_2, ORANGE, style_axes
from utils import load_scaler, load_split

log = get_logger("train")


def build_model() -> keras.Model:
    """1D-CNN regression model: [batch, 24, 4] -> [batch, 24]."""
    inputs = keras.Input(shape=(SEQ_LEN, N_FEATURES), name="sensor_window")
    x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Conv1D(8, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    outputs = layers.Dense(HORIZON, name="inflow_forecast")(x)   # linear output
    return keras.Model(inputs, outputs, name="inflow_forecaster")


class EpochLogger(keras.callbacks.Callback):
    """Compact progress log: first epoch, then every 5th epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch == 0 or (epoch + 1) % 5 == 0:
            log.info("epoch %3d | loss %.6f | val_loss %.6f | val_mae %.5f | lr %.1e",
                     epoch + 1, logs.get("loss", float("nan")), logs.get("val_loss", float("nan")),
                     logs.get("val_mae", float("nan")),
                     float(keras.backend.get_value(self.model.optimizer.learning_rate)))


def plot_curves(history: dict) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(epochs, history["loss"], color=BLUE, linewidth=2, label="Training loss")
    ax.plot(epochs, history["val_loss"], color=ORANGE, linewidth=2, label="Validation loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised units, log scale)")
    ax.set_title("Training curves", loc="left", fontsize=12, color="#0b0b0b")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(TRAIN_CURVES_PNG, dpi=150)
    plt.close(fig)


def run(epochs: int = 150, batch_size: int = 128, learning_rate: float = 1e-3,
        patience: int = 12, seed: int = DEFAULT_SEED) -> dict:
    ensure_dirs()
    seed_everything(seed)
    keras.utils.set_random_seed(seed)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    log.info("Train %s -> %s | Val %s -> %s", x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    model = build_model()
    model.summary(print_fn=lambda line: log.info(line))
    n_params = model.count_params()
    log.info("Parameters: %d (~%.1f KB as int8 weights)", n_params, n_params / 1024)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=["mae"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 3),
                                          min_lr=1e-5),
        EpochLogger(),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                        batch_size=batch_size, shuffle=True, callbacks=callbacks, verbose=0)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best = int(np.argmin(hist["val_loss"]))
    scaler = load_scaler()
    val_mae_m3s = hist["val_mae"][best] * scaler.target_span
    log.info("Stopped after %d epochs; best epoch %d | val_loss %.6f | val MAE %.4f m3/s",
             len(hist["loss"]), best + 1, hist["val_loss"][best], val_mae_m3s)

    model.save(KERAS_MODEL_PATH)
    TRAIN_HISTORY_JSON.write_text(json.dumps(hist, indent=2))
    plot_curves(hist)
    log.info("Saved %s", KERAS_MODEL_PATH)
    return {"epochs_run": len(hist["loss"]), "best_epoch": best + 1,
            "best_val_loss": hist["val_loss"][best], "val_mae_m3_s": val_mae_m3s, "parameters": n_params}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    run(a.epochs, a.batch_size, a.lr, a.patience, a.seed)
