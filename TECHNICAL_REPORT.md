# Technical Report: Intelligent Flood Monitoring and Outlet Control

## 1. Problem

Flood risk can increase rapidly while outlet debris, poor communications and
incomplete measurements prevent a safe or timely response. A fixed threshold
controller may react too late, cannot anticipate short-term basin demand, and
may assume that an outlet is usable even when plastic or rock obstructs it.
Operators therefore need basin-level forecasting, continuous visual inspection,
and a controlled release response that remains useful when network access is
intermittent.

The problem is not limited to a laboratory demonstration. A field system must
monitor rainfall, basin volume, discharge and soil condition; assess hydrological
risk; detect debris at the outlet; and produce a verifiable mechanical response.
It must also define fallback behaviour for invalid sensor or model results. The
project addresses these requirements with local edge processing and independent
hydrological, visual and actuator layers.

## 2. Proposed Solution

The proposed system combines a lightweight inflow forecast, debris-aware computer
vision and an adaptive outlet policy. It is designed for a stormwater basin or
urban flood-control structure with local sensing, an edge controller and an
optional supervisory connection.

Rainfall, basin level, outlet discharge and soil saturation provide a 24-sample,
five-minute hydrological window. The ESP32 executes a quantized TensorFlow Lite
model and predicts 24 future inflow values, providing a two-hour horizon. The decision logic combines
this horizon with measured basin volume and conservative forecast scenarios.
It prefers 50% basin capacity, progressively releases water above this level,
and uses a full-opening emergency action at or above 70%. The smallest valve
opening that maintains an acceptable projected state is selected, with a deadband
to prevent unnecessary movement.

An independent camera connected to a Raspberry Pi monitors the outlet approach.
YOLOv8 running on the Raspberry Pi detects plastic and rock debris, while OpenCV
measures detection area, location and severity. Blocking conditions generate
CLEAR, WARNING_DEBRIS or ALERT_BLOCKAGE results. Vision is deliberately not the
sole flood-gate authority: it provides maintenance and supervisory alerts, which
are combined with measured basin state and validated hydraulic logic.

A prototype implementation demonstrates the complete control chain. Python tools
generate storm data, train and quantize the model, and simulate basin control.
An ESP32 runs full-INT8 TensorFlow Lite Micro inference and controls two servos
through PWM. It also uses an ultrasonic level sensor and analogue sensor inputs.
A Raspberry Pi runs the CV implementation on recorded video or a webcam and
records detections and JSONL/MQTT alerts.

A single Raspberry Pi could technically combine both roles, but the proposed
controller uses the ESP32 for real-time sensor acquisition and servo control,
while reserving the Pi for computer vision. The ESP32 provides a fast, isolated,
low-latency control loop that remains responsive even if the Pi or network is
busy or unavailable. The Pi is therefore used for computationally heavier image
analysis rather than safety-critical real-time actuation.

## 3. System Architecture

```text
Rainfall | basin level | discharge | soil sensors
                         |
                         v
                    ESP32 controller
       acquisition, validation and local control
                         |
          +--------------+--------------+
          |                             |
          v                             v
24 x 4 inflow window            Raspberry Pi
          |                    YOLOv8 CV + camera
          v                             |
INT8 forecast + projection             v
          |                       debris status
          +--------------+--------------+
                         |
                         v
               Servo gate actuator

ESP32 and Raspberry Pi communicate by serial/MQTT as optional status integration.
```

The ESP32 reads four sensor channels, maintains the forecasting window and
executes the embedded inference and local mechanical control. In the reference
prototype, GPIO 25 and GPIO 26 drive independent servos with common ground and
an external 5--6 V regulated servo supply. Servo positive supplies must not be
connected to the ESP32 3.3 V pin.

The CV process is deployed on the Raspberry Pi, while the ESP32 remains
responsible for local hydrological protection and gate control. The Raspberry
Pi sends blockage status to the ESP32 or supervisory system over serial, JSONL
or MQTT. This separation keeps real-time sensing and servo timing independent
of camera processing and general-purpose networking. In a field deployment, a
higher-level controller or trained operator should confirm the final release
command, particularly for an obstruction or sensor fault.

## 4. Main Technologies

- **Hydrological intelligence:** Python, NumPy, TensorFlow/Keras, TensorFlow
  Lite and TensorFlow Lite for Microcontrollers.
- **Computer vision node:** Raspberry Pi, Python, Ultralytics YOLOv8,
  OpenCV and NumPy.
- **Embedded platform:** ESP32, Arduino framework, PlatformIO, ESP32Servo,
  ADC inputs and ultrasonic level sensing.
- **Mechanical control:** Two PWM servos, external regulated servo power,
  independent calibration and deadband control.
- **Integration:** Serial diagnostics, JSONL detection logs and optional MQTT
  alerts. MQTT or cloud connectivity is optional; safe local decisions do not
  depend on continuous network access.
- **Evaluation:** Synthetic storm episodes, storm-level data separation,
  held-out model metrics and offline basin mass-balance simulation.

## 5. Intelligent Processing Approach

The synthetic catchment generator combines rainfall, Horton infiltration,
soil saturation, runoff routing and basin storage. The 1D-CNN receives 24
five-minute observations, each containing rainfall, basin volume, discharge
and soil saturation. It predicts 24 future inflow values.

The full-INT8 TensorFlow Lite model is 11.32 KB and has approximately 4.5 thousand
parameters. It uses RESHAPE, CONV2D, MAX_POOL2D and FULLY_CONNECTED operations.
On 6,570 held-out synthetic windows, INT8 achieved MAE 0.0768 m^3/s and RMSE
0.1998 m^3/s, versus float32 MAE 0.0739 m^3/s and RMSE 0.1889 m^3/s. These values
measure synthetic forecast accuracy and are not a field-hydrology guarantee.

The offline controller evaluates forecast scenarios of 80%, 100%, 125% and
150%, then projects basin volume for each valve opening. The result is the
lowest opening that keeps the worst projected state below the emergency threshold.
In a ten-episode test, zero overflow events occurred in seven physically feasible
storms. Three storms were identified as physically infeasible because full outlet
capacity could not prevent the basin from reaching 100%.

The CV dataset pipeline combines selected YOLOv8 source datasets and remaps
annotations to `rock` and `plastic`. Training uses YOLOv8s at 640-pixel resolution
for 60 epochs. Runtime detection confidence is 0.15. OpenCV computes the
percentage of frame coverage for each object, identifies left/centre/right and
top/middle/bottom position, and marks whether the object lies in the central
approach path.

Severity is LOW, MEDIUM or HIGH. The edge decision is CLEAR for no detections,
WARNING_DEBRIS for detections without a high-severity result, and ALERT_BLOCKAGE
for high severity or combined coverage above 15% of the frame. `debris_edge.py`
draws boxes and status, saves annotated video, records JSONL events and can
publish MQTT alerts at five-second intervals. `detect_labels.py` provides simpler
plastic and rock visualisation for live or recorded video.

## 6. Innovation

The innovation is the integration of four functions that are often handled
separately: local inflow forecasting, basin-state projection, debris-aware visual
assessment and adaptive mechanical control. A two-node controller combines a
Raspberry Pi for computer-vision processing with an ESP32 for deterministic
sensor acquisition and real-time gate control. Conservative scenario analysis
reduces dependence on a single point forecast, while the small INT8 model allows
hydrological inference near the site. Independent servo calibration supports
different gate linkages, and failure handling prevents a lost level measurement
from being interpreted as a safe level.

The architecture is also modular. Hydrological and visual decisions can be
validated independently, then fused as a supervisory recommendation. The vision
subsystem contributes maintenance visibility without becoming an unsafe direct
gate actuator. JSONL and MQTT status signals provide an auditable integration
path for an operator dashboard or supervisory controller.

## 7. Expected Impact

Validated in a real catchment, the system could provide earlier flood-risk
warning, reduce dependence on continuous cloud connectivity and help maintenance
teams identify blocked outlets before discharge assumptions become hazardous.
Each basin would retain local sensing and processing while an optional dashboard
could aggregate basin status, debris alerts and operator decisions.

The solution can scale to multiple basins because each site has an independent
edge decision layer. Replacing the synthetic inputs and recalibrating hydraulic
parameters would adapt the concept to local catchments. The expected impact
includes fewer manual inspections, faster maintenance response, more informed
outlet operation and a reusable template for resilient smart infrastructure.

The prototype is not yet a certified flood-control system. Synthetic data does
not prove real-hydraulic performance, the demonstration model and servos do not
establish full-scale gate safety, and no field CV accuracy or latency is claimed.
Deployment would require site-specific calibration, representative camera data,
mechanical sizing, fail-safe power, outage testing and independent safety review.
