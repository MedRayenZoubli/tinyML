"""Full-INT8 post-training quantization and C++ export (pipeline step 3).

* Converts the trained Keras model with ``tf.lite.TFLiteConverter`` to a fully
  integer model (INT8 weights, activations *and* int8 input/output tensors).
* Calibrates activation ranges with a 100-sample representative dataset, spread
  evenly over the forecast-peak distribution (dry -> the single stormiest window).
  Calibrating on random windows alone would clip the hydrograph peaks.
* Writes ``inflow_forecast_model.tflite``, ``model.h`` (g_model / g_model_len),
  ``scaler_params.h`` (normalisation + quantisation constants for the firmware)
  and ``export_info.json``.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    DEFAULT_SEED,
    EXPORT_INFO_JSON,
    HORIZON,
    KERAS_MODEL_PATH,
    MODEL_HEADER_PATH,
    N_FEATURES,
    SCALER_HEADER_PATH,
    SEQ_LEN,
    TENSOR_ARENA_BUDGET_BYTES,
    TFLITE_MODEL_PATH,
    ensure_dirs,
    get_logger,
)
from utils import load_scaler, load_split, make_interpreter

log = get_logger("quantize")

# TFLM resolver registration call for each builtin op the converted graph may contain.
TFLM_RESOLVER_CALL = {
    "CONV_2D": "AddConv2D", "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D", "MAX_POOL_2D": "AddMaxPool2D",
    "FULLY_CONNECTED": "AddFullyConnected", "RESHAPE": "AddReshape", "EXPAND_DIMS": "AddExpandDims",
    "SQUEEZE": "AddSqueeze", "QUANTIZE": "AddQuantize", "DEQUANTIZE": "AddDequantize",
    "PAD": "AddPad", "STRIDED_SLICE": "AddStridedSlice", "SHAPE": "AddShape", "PACK": "AddPack",
    "CONCATENATION": "AddConcatenation", "RELU": "AddRelu", "TRANSPOSE": "AddTranspose",
}
DTYPE_BYTES = {"FLOAT32": 4, "INT32": 4, "INT8": 1, "UINT8": 1, "INT16": 2, "INT64": 8, "BOOL": 1, "FLOAT16": 2}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def select_calibration_indices(y_train: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n`` windows at evenly spaced ranks of their forecast peak, dry -> extreme.

    The first and last ranks are the quietest and the single stormiest window, so the
    calibrated int8 output range always covers the largest hydrograph peak in the
    training data. (A purely random sample is mostly dry weather and clips the peaks.)
    """
    order = np.argsort(y_train.max(axis=1))
    ranks = np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))
    idx = order[ranks]
    rng.shuffle(idx)
    return idx


def make_representative_dataset(x_train: np.ndarray, indices: np.ndarray):
    def representative_dataset():
        for i in indices:
            yield [x_train[i:i + 1].astype(np.float32)]   # batch of 1, float32
    return representative_dataset


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def _configure(converter: tf.lite.TFLiteConverter, representative_dataset) -> None:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8


def with_fixed_batch(model: keras.Model) -> keras.Model:
    """Rebuild the model with batch size 1.

    A dynamic batch dimension makes Flatten lower to SHAPE + STRIDED_SLICE + PACK +
    RESHAPE. With a fixed batch it becomes a single constant RESHAPE, so the ESP32
    firmware needs fewer ops registered.
    """
    inp = keras.Input(shape=(SEQ_LEN, N_FEATURES), batch_size=1, name="sensor_window")
    return keras.Model(inp, model(inp), name=model.name)


def convert_to_int8(model: keras.Model, representative_dataset) -> bytes:
    model = with_fixed_batch(model)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        _configure(converter, representative_dataset)
        return converter.convert()
    except Exception as exc:  # Keras-3 / TF version quirks: retry through a SavedModel export
        log.warning("from_keras_model failed (%s); retrying via SavedModel export", exc)
        saved_dir = TFLITE_MODEL_PATH.parent / "_saved_model_tmp"
        model.export(str(saved_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
        _configure(converter, representative_dataset)
        return converter.convert()


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def inspect_model(tflite_bytes: bytes, interpreter) -> dict:
    """List the builtin ops (for the TFLM resolver) and bound the arena size.

    The arena bound sums every non-constant tensor with *no* buffer reuse, so it
    over-estimates activation memory. TFLM adds a few KB of bookkeeping on top;
    confirm on-device with ``interpreter.arena_used_bytes()``.
    """
    try:
        from tensorflow.lite.python import schema_py_generated as fb

        model = fb.ModelT.InitFromObj(fb.Model.GetRootAsModel(bytearray(tflite_bytes), 0))
        graph = model.subgraphs[0]
        op_names = {v: k for k, v in vars(fb.BuiltinOperator).items() if isinstance(v, int)}
        type_names = {v: k for k, v in vars(fb.TensorType).items() if isinstance(v, int)}
        ops = []
        for op in graph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
            ops.append(op_names.get(builtin, f"UNKNOWN_{builtin}"))
        activation_bytes = 0
        for t in graph.tensors:
            buf = model.buffers[t.buffer].data
            if buf is not None and len(buf) > 0:
                continue  # constant (weights/bias): lives in flash, not in the arena
            n_elem = int(np.prod([max(int(d), 1) for d in t.shape])) if t.shape is not None else 1
            activation_bytes += n_elem * DTYPE_BYTES.get(type_names.get(t.type, ""), 4)
    except Exception as exc:  # fall back to the interpreter's own op list
        log.warning("Flatbuffer inspection unavailable (%s); using interpreter op list", exc)
        ops = [d["op_name"] for d in interpreter._get_ops_details()]
        activation_bytes = None

    unique_ops = list(dict.fromkeys(ops))
    return {"ops_in_order": ops, "unique_ops": unique_ops,
            "resolver_calls": [TFLM_RESOLVER_CALL.get(o, f"/* {o}: see TFLM docs */") for o in unique_ops],
            "activation_bytes_upper_bound": activation_bytes}


# --------------------------------------------------------------------------- #
# C++ export
# --------------------------------------------------------------------------- #
def _float_literal(v: float) -> str:
    s = f"{float(v):.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def write_model_header(data: bytes, info: dict) -> None:
    rows = []
    for i in range(0, len(data), 12):
        rows.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]))
    text = (
        "// Auto-generated by quantize_and_export.py -- do not edit.\n"
        f"// Full-INT8 TFLite model: input [1, {SEQ_LEN}, {N_FEATURES}] int8 -> output [1, {HORIZON}] int8.\n"
        f"// Input  quantization: scale={info['input_scale']:.9g} zero_point={info['input_zero_point']}\n"
        f"// Output quantization: scale={info['output_scale']:.9g} zero_point={info['output_zero_point']}\n"
        f"// Ops for MicroMutableOpResolver: {', '.join(info['inspection']['unique_ops'])}\n"
        "#ifndef INFLOW_FORECAST_MODEL_H_\n"
        "#define INFLOW_FORECAST_MODEL_H_\n\n"
        "alignas(8) const unsigned char g_model[] = {\n"
        + ",\n".join(rows) + "\n};\n\n"
        f"const unsigned int g_model_len = {len(data)};\n\n"
        "#endif  // INFLOW_FORECAST_MODEL_H_\n"
    )
    MODEL_HEADER_PATH.write_text(text)


def write_scaler_header(scaler, info: dict) -> None:
    fmin = ", ".join(_float_literal(v) for v in scaler.feature_min)
    fmax = ", ".join(_float_literal(v) for v in scaler.feature_max)
    text = (
        "// Auto-generated by quantize_and_export.py -- do not edit.\n"
        "// Normalisation (fitted on the training split) and int8 I/O quantization constants.\n"
        "//   x_norm = (x - kFeatureMin[c]) / (kFeatureMax[c] - kFeatureMin[c])\n"
        "//   x_q    = clamp(round(x_norm / kInputScale) + kInputZeroPoint, -128, 127)\n"
        "//   y_norm = (y_q - kOutputZeroPoint) * kOutputScale\n"
        "//   inflow_m3_s = y_norm * (kTargetMax - kTargetMin) + kTargetMin\n"
        "// Channel order: precipitation mm/hr, basin volume m3, discharge m3/s, soil saturation %.\n"
        "#ifndef INFLOW_SCALER_PARAMS_H_\n"
        "#define INFLOW_SCALER_PARAMS_H_\n\n"
        f"constexpr int kSeqLen = {SEQ_LEN};\n"
        f"constexpr int kNumFeatures = {N_FEATURES};\n"
        f"constexpr int kHorizon = {HORIZON};\n\n"
        f"constexpr float kFeatureMin[kNumFeatures] = {{{fmin}}};\n"
        f"constexpr float kFeatureMax[kNumFeatures] = {{{fmax}}};\n"
        f"constexpr float kTargetMin = {_float_literal(scaler.target_min)};\n"
        f"constexpr float kTargetMax = {_float_literal(scaler.target_max)};\n\n"
        f"constexpr float kInputScale = {_float_literal(info['input_scale'])};\n"
        f"constexpr int kInputZeroPoint = {info['input_zero_point']};\n"
        f"constexpr float kOutputScale = {_float_literal(info['output_scale'])};\n"
        f"constexpr int kOutputZeroPoint = {info['output_zero_point']};\n\n"
        "#endif  // INFLOW_SCALER_PARAMS_H_\n"
    )
    SCALER_HEADER_PATH.write_text(text)


# --------------------------------------------------------------------------- #
def run(n_calibration: int = 100, seed: int = DEFAULT_SEED) -> dict:
    ensure_dirs()
    rng = np.random.default_rng(seed)

    model = keras.models.load_model(KERAS_MODEL_PATH)
    x_train, y_train = load_split("train")
    scaler = load_scaler()

    idx = select_calibration_indices(y_train, n_calibration, rng)
    log.info("Calibrating with %d representative samples (evenly spaced by forecast-peak rank, incl. the max storm)",
             len(idx))
    tflite_bytes = convert_to_int8(model, make_representative_dataset(x_train, idx))
    TFLITE_MODEL_PATH.write_bytes(tflite_bytes)
    log.info("Saved %s (%d bytes)", TFLITE_MODEL_PATH, len(tflite_bytes))

    interpreter = make_interpreter(model_content=tflite_bytes)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8, "model I/O is not int8"
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    log.info("Input  int8 %s | scale %.6g zero_point %d", [int(d) for d in inp["shape"]], in_scale, in_zp)
    log.info("Output int8 %s | scale %.6g zero_point %d", [int(d) for d in out["shape"]], out_scale, out_zp)

    inspection = inspect_model(tflite_bytes, interpreter)
    log.info("Ops used: %s", ", ".join(inspection["unique_ops"]))
    log.info("TFLM resolver: tflite::MicroMutableOpResolver<%d> with %s",
             len(inspection["unique_ops"]), ", ".join(inspection["resolver_calls"]))
    bound = inspection["activation_bytes_upper_bound"]
    if bound is not None:
        log.info("Activation tensors (no reuse, upper bound): %d bytes vs %d byte arena budget (%.1f%%)",
                 bound, TENSOR_ARENA_BUDGET_BYTES, 100 * bound / TENSOR_ARENA_BUDGET_BYTES)

    info = {"tflite_bytes": len(tflite_bytes), "input_scale": float(in_scale), "input_zero_point": int(in_zp),
            "output_scale": float(out_scale), "output_zero_point": int(out_zp),
            "calibration_samples": int(len(idx)), "tensor_arena_budget_bytes": TENSOR_ARENA_BUDGET_BYTES,
            "inspection": inspection}
    write_model_header(tflite_bytes, info)
    write_scaler_header(scaler, info)
    EXPORT_INFO_JSON.write_text(json.dumps(info, indent=2))
    log.info("Wrote %s and %s", MODEL_HEADER_PATH, SCALER_HEADER_PATH)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    run(a.calibration_samples, a.seed)
