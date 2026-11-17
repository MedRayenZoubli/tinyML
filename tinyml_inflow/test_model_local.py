"""Local accuracy test of the INT8 TFLite model (pipeline step 4).

Mirrors what the ESP32 does: quantize float inputs to int8 with the model's own
(scale, zero_point), run the interpreter, de-quantize the int8 outputs. Compares

    actual inflow   vs   float32 Keras baseline   vs   INT8 TFLite

in physical units (m3/s) and plots one 2-hour storm forecast.
Outputs (artifacts/): metrics.json, inflow_forecast_comparison.png
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from config import (
    COMPARISON_PNG,
    FEATURE_NAMES,
    HORIZON,
    KERAS_MODEL_PATH,
    METRICS_JSON,
    STEP_MINUTES,
    TFLITE_MODEL_PATH,
    ensure_dirs,
    get_logger,
)
from plot_style import BLUE, INK, INK_2, MUTED, ORANGE, SURFACE, style_axes
from utils import load_scaler, load_split, make_interpreter

log = get_logger("test")


def quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """float32 -> int8:  q = round(x / scale) + zero_point, clipped to [-128, 127]."""
    return np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)


def dequantize_output(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """int8 -> float32:  x = (q - zero_point) * scale."""
    return (q.astype(np.float32) - zero_point) * scale


def run_tflite_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Run every sample (batch size 1, like the MCU) through the INT8 interpreter."""
    interpreter = make_interpreter(model_path=TFLITE_MODEL_PATH)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    log.info("INT8 I/O params | input scale %.6g zp %d | output scale %.6g zp %d",
             in_scale, in_zp, out_scale, out_zp)

    preds = np.empty((len(x), HORIZON), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(len(x)):
        interpreter.set_tensor(inp["index"], quantize_input(x[i:i + 1], in_scale, in_zp))
        interpreter.invoke()
        preds[i] = dequantize_output(interpreter.get_tensor(out["index"])[0], out_scale, out_zp)
    per_sample_us = (time.perf_counter() - t0) / len(x) * 1e6
    return preds, per_sample_us


def simulate_stream(
    x: np.ndarray,
    y_norm: np.ndarray,
    scaler,
    sample_count: int,
    start_index: int = 0,
    interval_seconds: float = 0.0,
) -> None:
    """Replay test windows one at a time as if sensor data were arriving live."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if start_index < 0 or start_index >= len(x):
        raise ValueError(f"start_index must be between 0 and {len(x) - 1}.")
    end_index = min(start_index + sample_count, len(x))

    interpreter = make_interpreter(model_path=TFLITE_MODEL_PATH)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    print("\nStreaming sensor windows into the INT8 model")
    print(f"Samples {start_index}-{end_index - 1}; interval {interval_seconds:.2f}s")
    print("Forecast values are the next 2 hours at 5-minute intervals.\n")

    for sample_index in range(start_index, end_index):
        window = x[sample_index:sample_index + 1]
        interpreter.set_tensor(inp["index"], quantize_input(window, in_scale, in_zp))
        interpreter.invoke()
        pred_norm = dequantize_output(interpreter.get_tensor(out["index"])[0], out_scale, out_zp)
        prediction = scaler.denormalize_target(pred_norm)
        actual = scaler.denormalize_target(y_norm[sample_index])
        raw_latest = window[0, -1] * scaler.feature_span + scaler.feature_min
        input_values = {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, raw_latest)
        }
        error = float(np.mean(np.abs(prediction - actual)))

        print(
            f"[window {sample_index:5d}] "
            f"latest sensors={input_values} | "
            f"forecast first={prediction[0]:.3f} m3/s | "
            f"peak={prediction.max():.3f} m3/s at +{int(prediction.argmax() + 1) * STEP_MINUTES} min | "
            f"actual peak={actual.max():.3f} m3/s | "
            f"MAE={error:.3f}"
        )
        print("  forecast:", " ".join(f"{value:.3f}" for value in prediction))
        if sample_index < end_index - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)


def regression_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - truth
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((truth - truth.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "nse": 1.0 - sse / sst if sst > 0 else float("nan")}


def pick_storm_sample(y_true: np.ndarray, y_float: np.ndarray) -> int:
    """A *typical* storm window for the plot (neither best nor worst case).

    Candidates are windows whose hydrograph peaks inside the 2 h horizon (not at its
    edges) with a peak above 0.5 m3/s. Among them, pick the one whose float32 MAE is
    the median, so the figure is not cherry-picked in either direction.
    """
    peak_idx, peak_val = y_true.argmax(axis=1), y_true.max(axis=1)
    cand = np.where((peak_idx >= 4) & (peak_idx <= 19) & (peak_val > 0.5))[0]
    if len(cand) == 0:
        return int(np.argmax(peak_val))
    mae = np.mean(np.abs(y_float[cand] - y_true[cand]), axis=1)
    return int(cand[np.argsort(mae)[len(cand) // 2]])


def plot_comparison(y_true, y_float, y_int8, sample: int, summary: dict, n_samples: int) -> None:
    t = np.arange(1, HORIZON + 1) * STEP_MINUTES            # minutes ahead
    a, f, q = y_true[sample], y_float[sample], y_int8[sample]

    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.12})
    ax.plot(t, a, color=INK, linewidth=2.4, label="Actual inflow")
    ax.plot(t, f, color=BLUE, linewidth=2, label="Float32 Keras model")
    ax.plot(t, q, color=ORANGE, linewidth=2, linestyle=(0, (5, 2)), marker="o", markevery=2,
            markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1, label="INT8 TFLite model")
    ax.set_ylabel("Inflow (m³/s)")
    ax.set_ylim(bottom=0)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper left")

    ax_err.axhline(0, color=MUTED, linewidth=0.8)
    ax_err.plot(t, f - a, color=BLUE, linewidth=2)
    ax_err.plot(t, q - a, color=ORANGE, linewidth=2, linestyle=(0, (5, 2)))
    ax_err.set_ylabel("Error vs actual (m³/s)")
    ax_err.set_xlabel("Minutes ahead (2-hour forecast horizon, 5-minute steps)")
    ax_err.set_xlim(t[0], t[-1])
    ax_err.set_xticks(np.arange(15, 121, 15))
    style_axes(ax_err)

    s = summary
    fig.suptitle("2-hour storm inflow forecast: actual vs float32 vs INT8", x=0.07, y=0.98,
                 ha="left", fontsize=13, color=INK)
    fig.text(0.07, 0.915, "Plotted window: typical storm (median float32 error among storm windows)",
             ha="left", fontsize=9, color=INK_2)
    fig.text(0.07, 0.888,
             f"Whole test set ({n_samples} windows): MAE float32 {s['float32_vs_actual']['mae']:.4f} · "
             f"INT8 {s['int8_vs_actual']['mae']:.4f} m³/s · INT8 vs float32 MAE "
             f"{s['int8_vs_float32']['mae']:.4f}, max {s['int8_vs_float32']['max_abs_error']:.2f} m³/s",
             ha="left", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.855, bottom=0.09)
    fig.savefig(COMPARISON_PNG, dpi=150)
    plt.close(fig)


def run(max_samples: int | None = None) -> dict:
    ensure_dirs()
    scaler = load_scaler()
    x, y_norm = load_split("test")
    if max_samples and len(x) > max_samples:
        keep = np.linspace(0, len(x) - 1, max_samples).astype(int)   # evenly spaced across episodes
        x, y_norm = x[keep], y_norm[keep]
    log.info("Test samples: %d  X%s  y%s", len(x), x.shape, y_norm.shape)

    # Float32 baseline
    model = keras.models.load_model(KERAS_MODEL_PATH)
    pred_float_norm = model.predict(x, batch_size=512, verbose=0)

    # INT8 TFLite
    log.info("Running INT8 TFLite inference sample by sample ...")
    pred_int8_norm, per_sample_us = run_tflite_int8(x)
    log.info("Host interpreter speed: %.0f us per inference (desktop CPU; not ESP32 timing)", per_sample_us)

    # Back to physical units (m3/s)
    y_true = scaler.denormalize_target(y_norm)
    y_float = scaler.denormalize_target(pred_float_norm)
    y_int8 = scaler.denormalize_target(pred_int8_norm)

    f_vs_a = regression_metrics(y_true, y_float)
    q_vs_a = regression_metrics(y_true, y_int8)
    q_err = y_int8 - y_float
    quant = {"mae": float(np.mean(np.abs(q_err))), "rmse": float(np.sqrt(np.mean(q_err ** 2))),
             "max_abs_error": float(np.max(np.abs(q_err))),
             "pct_of_target_range": float(100 * np.mean(np.abs(q_err)) / scaler.target_span)}

    summary = {
        "samples": int(len(x)),
        "units": "m3/s",
        "float32_vs_actual": f_vs_a,
        "int8_vs_actual": q_vs_a,
        "int8_vs_float32": quant,
        "mae_increase_from_quantization": q_vs_a["mae"] - f_vs_a["mae"],
        "rmse_increase_from_quantization": q_vs_a["rmse"] - f_vs_a["rmse"],
        "mae_by_horizon_step_float32": [float(v) for v in np.mean(np.abs(y_float - y_true), axis=0)],
        "mae_by_horizon_step_int8": [float(v) for v in np.mean(np.abs(y_int8 - y_true), axis=0)],
        "host_inference_us_per_sample": per_sample_us,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=4))

    log.info("%-22s %10s %10s %10s", "(m3/s)", "MAE", "RMSE", "NSE")
    log.info("%-22s %10.5f %10.5f %10.4f", "float32 vs actual", f_vs_a["mae"], f_vs_a["rmse"], f_vs_a["nse"])
    log.info("%-22s %10.5f %10.5f %10.4f", "INT8 vs actual", q_vs_a["mae"], q_vs_a["rmse"], q_vs_a["nse"])
    log.info("%-22s %10.5f %10.5f   max |d| %.5f  (%.3f%% of target range)", "INT8 vs float32",
             quant["mae"], quant["rmse"], quant["max_abs_error"], quant["pct_of_target_range"])

    sample = pick_storm_sample(y_true, y_float)
    plot_comparison(y_true, y_float, y_int8, sample, summary, len(x))
    log.info("Saved %s (plotted test window #%d, peak %.3f m3/s)", COMPARISON_PNG, sample, y_true[sample].max())
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-samples", type=int, default=None, help="limit the number of test windows")
    ap.add_argument("--simulate", action="store_true",
                    help="replay test windows through the INT8 model one at a time")
    ap.add_argument("--simulation-samples", type=int, default=10,
                    help="number of windows to replay with --simulate")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first test-window index to replay with --simulate")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds to wait between replayed windows")
    a = ap.parse_args()
    if a.simulate:
        scaler = load_scaler()
        x, y_norm = load_split("test")
        simulate_stream(x, y_norm, scaler, a.simulation_samples,
                        a.start_index, a.interval)
    else:
        run(a.max_samples)
