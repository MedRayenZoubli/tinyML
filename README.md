# TinyML inflow forecaster for ESP32 (INT8, TensorFlow Lite for Microcontrollers)

Trains a ~4.5 k-parameter 1D-CNN that forecasts basin inflow for the next 2 hours from the
last 2 hours of sensor data, quantizes it to full INT8, and exports it as a C++ header.

| | Shape | Content |
|---|---|---|
| Input | `[batch, 24, 4]` | 24 five-minute steps x (precipitation mm/hr, basin volume m³, discharge m³/s, soil saturation %) |
| Output | `[batch, 24]` | inflow in m³/s for the 24 steps *after* the input window |

## Run

```bash
pip install -r requirements.txt
python main.py                 # generate -> train -> quantize/export -> local test
python main.py --epochs 40     # quicker run
python main.py --stages test   # re-run selected stages only
python test_model_local.py --simulate --simulation-samples 10 --interval 1
python controller_simulation.py --episodes 3 --steps 192
```

Each stage is also runnable on its own (`python generate_dataset.py`, `python train_model.py`, ...).

## Layout

| File | Role |
|---|---|
| `config.py` | Paths, tensor geometry, logging (single source of truth) |
| `generate_dataset.py` | Physical storm simulator -> normalised `.npy` splits + scaler |
| `train_model.py` | Keras 1D-CNN, Adam + MSE, early stopping |
| `quantize_and_export.py` | Full-INT8 conversion, `model.h`, `scaler_params.h` |
| `test_model_local.py` | INT8 vs float32 vs actual: MAE/RMSE, quantization error, plot, and stream replay |
| `controller_simulation.py` | Offline closed-loop INT8 forecast and anticipatory valve controller |
| `main.py` | Runs the pipeline and optional offline controller |
| `utils.py`, `plot_style.py` | Scaler/IO helpers, chart styling |

Outputs: `data/` (datasets, `scaler.npz/.json`) and `artifacts/` (`inflow_forecast_model.tflite`,
`model.h`, `scaler_params.h`, `metrics.json`, `inflow_forecast_comparison.png`, training curves).

### Replay sensor data through the model

After the artifacts and test dataset exist, replay test windows through the deployed INT8
model one at a time:

```bash
python test_model_local.py --simulate --simulation-samples 10 --interval 1
```

Each line shows the latest input sensor window values, the first predicted inflow, predicted
peak and timing, the actual peak from the held-out target, and the window MAE. The following
line prints all 24 predicted five-minute values. Use `--start-index` to choose another test
window and `--interval 0` for immediate replay. This is a desktop replay of recorded synthetic
windows; it does not connect to live hardware.

### Offline closed-loop controller

The controller prototype uses the INT8 inflow forecast to choose a simulated continuous
valve opening. Basin physics run every minute, while the model is queried every five minutes
using the 24 five-minute samples it was trained on. The controller expands the point forecast
into multiple inflow scenarios (80%, 100%, 125%, and 150%), then selects the smallest opening
whose worst-case projected two-hour volume remains below 70% capacity. A 5% valve deadband
reduces unnecessary movement, and a full-opening safety override applies at 70%.

```bash
python controller_simulation.py --episodes 3 --steps 192
# or after the normal pipeline:
python main.py --stages control --episodes 3 --steps 192
```

Results are written to `artifacts/controller_simulation.json`, including feasible versus
physically infeasible storms, forecast MAE, early-warning time, recovery time, discharge,
and valve decisions. This is an offline simulation only: the source inflow trace is synthetic,
and no real valve or hardware is controlled.

## Synthetic data model

Gaussian rain cells with embedded bursts -> Horton-style infiltration whose capacity decays as the
soil saturates -> impervious + pervious runoff -> gamma unit hydrograph -> basin inflow. The basin
integrates inflow minus a valve-controlled orifice outflow. Sensor noise is added to the inputs only.
Train/val/test are split **by storm record**, so no storm appears in two splits. Scaling constants
are fitted on the training split only.

## Results of the reference run (seed 42, 6,570 test windows, m³/s)

| | MAE | RMSE |
|---|---|---|
| Float32 vs actual | 0.0692 | 0.1877 |
| INT8 vs actual | 0.0713 | 0.1881 |
| INT8 vs float32 | 0.0107 | 0.0169 |

Model: 11.5 KB `.tflite`, 4 ops (`RESHAPE, CONV_2D, MAX_POOL_2D, FULLY_CONNECTED`), activation
tensors ~1 KB without buffer reuse. Numbers are for synthetic data and will change with your seed.

## Using it on the ESP32 (sketch, not compiled against TFLM here)

```cpp
#include "model.h"
#include "scaler_params.h"
// + TFLM headers: micro_interpreter.h, micro_mutable_op_resolver.h, schema_generated.h

alignas(16) static uint8_t tensor_arena[32 * 1024];

static tflite::MicroMutableOpResolver<4> resolver;
resolver.AddReshape();  resolver.AddConv2D();
resolver.AddMaxPool2D(); resolver.AddFullyConnected();

static tflite::MicroInterpreter interp(tflite::GetModel(g_model), resolver,
                                       tensor_arena, sizeof(tensor_arena));
interp.AllocateTensors();

// x[t][c]: raw sensor values, oldest step first, channel order as in the table above
for (int t = 0; t < kSeqLen; ++t)
  for (int c = 0; c < kNumFeatures; ++c) {
    float n = (x[t][c] - kFeatureMin[c]) / (kFeatureMax[c] - kFeatureMin[c]);
    n = fminf(fmaxf(n, 0.f), 1.f);                       // clamp: int8 input range is [0, 1]
    int q = (int)lroundf(n / kInputScale) + kInputZeroPoint;
    interp.input(0)->data.int8[t * kNumFeatures + c] = (int8_t)fminf(fmaxf(q, -128), 127);
  }
interp.Invoke();
for (int i = 0; i < kHorizon; ++i) {
  float n = (interp.output(0)->data.int8[i] - kOutputZeroPoint) * kOutputScale;
  float inflow_m3_s = n * (kTargetMax - kTargetMin) + kTargetMin;
}
```

Check `interp.arena_used_bytes()` on the device; the 32 KB budget has a large margin here.

## Things to know

* **Clamp inputs.** Min-max scaling uses training ranges; a reading above the training maximum
  falls outside the int8 input range. In the reference run this caused the largest INT8-vs-float
  deviation (0.41 m³/s, in 24 of 6,570 windows with out-of-range input).
* **Calibration matters.** The representative set is spread over the forecast-peak distribution
  and includes the stormiest window. A random sample is mostly dry weather and clipped the
  output peaks (max INT8 error 1.5 m³/s versus 0.41 m³/s now).
* **Forecast limits.** The model sees no future rain, so peaks caused by rain that has not fallen
  yet are under-predicted (visible in the plot). Retrain on real site data before relying on it.
* `tf.lite.Interpreter` is deprecated in recent TensorFlow; the code falls back to
  `ai_edge_litert` if it disappears.
