 #include <Arduino.h>
 #include <ESP32Servo.h>
 #include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "scaler_params.h"

namespace {

constexpr int kRainPin = 34;
constexpr int kVolumePin = 35;
constexpr int kDischargePin = 32;
constexpr int kSoilPin = 33;
constexpr int kUltrasonicTrigPin = 27;
constexpr int kUltrasonicEchoPin = 14;
constexpr int kGateServoAPin = 25;
constexpr int kGateServoBPin = 26;
constexpr unsigned long kSampleIntervalMs = 5000;
constexpr size_t kTensorArenaBytes = 96 * 1024;
constexpr float kBasinCapacityM3 = 4000.0f;
constexpr float kEmergencyVolumeM3 = kBasinCapacityM3 * 0.70f;
constexpr float kPreferredVolumeM3 = kBasinCapacityM3 * 0.50f;
constexpr float kGateDeadband = 0.05f;
constexpr int kGateServoAClosedAngle = 10;
constexpr int kGateServoAOpenAngle = 170;
constexpr int kGateServoBClosedAngle = 170;
constexpr int kGateServoBOpenAngle = 10;

alignas(16) uint8_t tensorArena[kTensorArenaBytes];
float sensorWindow[kSeqLen][kNumFeatures];
size_t samplesInWindow = 0;
unsigned long lastSampleMs = 0;

tflite::MicroErrorReporter errorReporter;
tflite::MicroMutableOpResolver<4> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
Servo gateServoA;
Servo gateServoB;
float previousGateOpening = 0.0f;
float lastValidVolumeM3 = 0.0f;
bool levelSensorValid = false;

float mapAdcToRange(int pin, float minimum, float maximum) {
  const int raw = analogRead(pin);
  return minimum + (maximum - minimum) * (static_cast<float>(raw) / 4095.0f);
}

bool readWaterVolume(float* volumeM3) {
  digitalWrite(kUltrasonicTrigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(kUltrasonicTrigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(kUltrasonicTrigPin, LOW);

  const unsigned long duration =
      pulseIn(kUltrasonicEchoPin, HIGH, 30000UL);
  if (duration == 0) {
    return false;
  }

  const float distanceCm = duration * 0.0343f / 2.0f;
  constexpr float kEmptyDistanceCm = 35.0f;
  constexpr float kFullDistanceCm = 5.0f;
  const float levelFraction = constrain(
      (kEmptyDistanceCm - distanceCm) /
          (kEmptyDistanceCm - kFullDistanceCm),
      0.0f, 1.0f);
  *volumeM3 = levelFraction * kBasinCapacityM3;
  return true;
}

bool readSensors(float values[kNumFeatures]) {
  values[0] = mapAdcToRange(kRainPin, kFeatureMin[0], kFeatureMax[0]);
  values[2] = mapAdcToRange(kDischargePin, kFeatureMin[2], kFeatureMax[2]);
  values[3] = mapAdcToRange(kSoilPin, kFeatureMin[3], kFeatureMax[3]);

#ifdef USE_ULTRASONIC
  if (!readWaterVolume(&values[1])) {
    values[1] = lastValidVolumeM3;
    levelSensorValid = false;
    return false;
  }
  lastValidVolumeM3 = values[1];
  levelSensorValid = true;
  return true;
#else
  values[1] = mapAdcToRange(kVolumePin, kFeatureMin[1], kFeatureMax[1]);
  lastValidVolumeM3 = values[1];
  levelSensorValid = true;
  return true;
#endif
}

int8_t quantizeInput(float value, int feature) {
  float normalized = (value - kFeatureMin[feature]) /
                     (kFeatureMax[feature] - kFeatureMin[feature]);
  normalized = constrain(normalized, 0.0f, 1.0f);
  const int quantized = lroundf(normalized / kInputScale) + kInputZeroPoint;
  return static_cast<int8_t>(constrain(quantized, -128, 127));
}

void appendSample(const float values[kNumFeatures]) {
  if (samplesInWindow < kSeqLen) {
    memcpy(sensorWindow[samplesInWindow], values,
           sizeof(sensorWindow[samplesInWindow]));
    ++samplesInWindow;
    return;
  }

  memmove(sensorWindow[0], sensorWindow[1],
          sizeof(sensorWindow[0]) * (kSeqLen - 1));
  memcpy(sensorWindow[kSeqLen - 1], values, sizeof(sensorWindow[0]));
}

void printSensors(const float values[kNumFeatures]) {
  Serial.printf("sensors rain=%.2f mm/hr volume=%.2f m3 discharge=%.3f m3/s soil=%.2f%%\n",
                values[0], values[1], values[2], values[3]);
}

float chooseGateOpening(float currentVolume, const float forecast[kHorizon],
                        const char** reason) {
  if (currentVolume >= kEmergencyVolumeM3) {
    *reason = "emergency_override";
    return 1.0f;
  }

  float peakForecast = forecast[0];
  for (int step = 1; step < kHorizon; ++step) {
    peakForecast = max(peakForecast, forecast[step]);
  }

  // Simulation policy: use the level as the primary command and let the
  // forecast add at most 25% opening, rather than saturating on a long
  // worst-case projection that the physical demo cannot reproduce.
  const float levelOpening =
      constrain((currentVolume - kPreferredVolumeM3) /
                    (kEmergencyVolumeM3 - kPreferredVolumeM3),
                0.0f, 1.0f);
  const float forecastOpening = constrain(peakForecast / kTargetMax, 0.0f, 1.0f) * 0.25f;
  const float candidate = constrain(levelOpening + forecastOpening, 0.0f, 1.0f);

  if (candidate >= 0.95f) {
    *reason = "maximum_release_required";
    return 1.0f;
  }
  if (fabsf(candidate - previousGateOpening) < kGateDeadband) {
    *reason = "deadband_hold";
    return previousGateOpening;
  }
  *reason = currentVolume < kPreferredVolumeM3
                ? "forecast_safe"
                : "anticipatory_release";
  return candidate;
}

void setGateOpening(float opening, const char* reason) {
  opening = constrain(opening, 0.0f, 1.0f);
  const int angleA = static_cast<int>(lroundf(
      kGateServoAClosedAngle +
      opening * (kGateServoAOpenAngle - kGateServoAClosedAngle)));
  const int angleB = static_cast<int>(lroundf(
      kGateServoBClosedAngle +
      opening * (kGateServoBOpenAngle - kGateServoBClosedAngle)));
  gateServoA.write(angleA);
  gateServoB.write(angleB);
  previousGateOpening = opening;
  Serial.printf("gate opening=%.0f%% servoA=%d servoB=%d reason=%s\n",
                opening * 100.0f, angleA, angleB, reason);
}

void runForecast() {
  for (int step = 0; step < kSeqLen; ++step) {
    for (int feature = 0; feature < kNumFeatures; ++feature) {
      input->data.int8[step * kNumFeatures + feature] =
          quantizeInput(sensorWindow[step][feature], feature);
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: model invocation failed");
    return;
  }

  float forecast[kHorizon];
  Serial.println("forecast (next 24 x 5-minute inflow values, m3/s):");
  for (int step = 0; step < kHorizon; ++step) {
    const float normalized =
        (static_cast<int>(output->data.int8[step]) - kOutputZeroPoint) *
        kOutputScale;
    forecast[step] =
        normalized * (kTargetMax - kTargetMin) + kTargetMin;
    Serial.printf("%02d: %.4f%s", step + 1, forecast[step],
                  (step % 4 == 3) ? "\n" : " | ");
  }

  const char* reason = nullptr;
  const float opening = levelSensorValid
                            ? chooseGateOpening(
                                  sensorWindow[kSeqLen - 1][1], forecast, &reason)
                            : 1.0f;
  if (!levelSensorValid) {
    reason = "level_sensor_failure";
  }
  setGateOpening(opening, reason);
}

bool initializeModel() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: incompatible TensorFlow Lite schema version");
    return false;
  }

  if (resolver.AddReshape() != kTfLiteOk ||
      resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("ERROR: failed to register model operators");
    return false;
  }

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, kTensorArenaBytes, &errorReporter);
  interpreter = &staticInterpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: tensor allocation failed");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input == nullptr || output == nullptr ||
      input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    Serial.println("ERROR: unexpected model tensor types");
    return false;
  }
  return true;
}

}

void setup() {
  Serial.begin(115200);
  delay(500);
  analogReadResolution(12);
#ifdef USE_ULTRASONIC
  pinMode(kUltrasonicTrigPin, OUTPUT);
  pinMode(kUltrasonicEchoPin, INPUT);
#endif
  gateServoA.attach(kGateServoAPin);
  gateServoB.attach(kGateServoBPin);
  setGateOpening(0.0f, "startup_warmup");

  Serial.println("\nTinyML inflow forecaster - Wokwi ESP32 simulation");
  if (!initializeModel()) {
    Serial.println("Model setup failed; stopping.");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("Turn the potentiometers to change the four simulated sensors.");
  Serial.println("Collecting the initial 24-sample sensor window...");
}

void loop() {
  const unsigned long now = millis();
  if (now - lastSampleMs < kSampleIntervalMs) {
    return;
  }
  lastSampleMs = now;

  float values[kNumFeatures];
  const bool levelValid = readSensors(values);
  appendSample(values);
  printSensors(values);
  if (!levelValid) {
    Serial.println("ERROR: ultrasonic level reading invalid; gate will fail open");
    setGateOpening(1.0f, "level_sensor_failure");
  }

  if (samplesInWindow == kSeqLen) {
    runForecast();
  } else {
    Serial.printf("window: %u/%d samples\n",
                  static_cast<unsigned>(samplesInWindow), kSeqLen);
  }
}
