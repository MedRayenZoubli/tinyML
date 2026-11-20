#include <Arduino.h>
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
constexpr unsigned long kSampleIntervalMs = 5000;
constexpr size_t kTensorArenaBytes = 32 * 1024;

alignas(16) uint8_t tensorArena[kTensorArenaBytes];
float sensorWindow[kSeqLen][kNumFeatures];
size_t samplesInWindow = 0;
unsigned long lastSampleMs = 0;

tflite::MicroErrorReporter errorReporter;
tflite::MicroMutableOpResolver<4> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float mapAdcToRange(int pin, float minimum, float maximum) {
  const int raw = analogRead(pin);
  return minimum + (maximum - minimum) * (static_cast<float>(raw) / 4095.0f);
}

void readSensors(float values[kNumFeatures]) {
  values[0] = mapAdcToRange(kRainPin, kFeatureMin[0], kFeatureMax[0]);
  values[1] = mapAdcToRange(kVolumePin, kFeatureMin[1], kFeatureMax[1]);
  values[2] = mapAdcToRange(kDischargePin, kFeatureMin[2], kFeatureMax[2]);
  values[3] = mapAdcToRange(kSoilPin, kFeatureMin[3], kFeatureMax[3]);
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

  Serial.println("forecast (next 24 x 5-minute inflow values, m3/s):");
  for (int step = 0; step < kHorizon; ++step) {
    const float normalized =
        (static_cast<int>(output->data.int8[step]) - kOutputZeroPoint) *
        kOutputScale;
    const float inflow =
        normalized * (kTargetMax - kTargetMin) + kTargetMin;
    Serial.printf("%02d: %.4f%s", step + 1, inflow,
                  (step % 4 == 3) ? "\n" : " | ");
  }
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
      model, resolver, tensorArena, kTensorArenaBytes, nullptr);
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

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);
  analogReadResolution(12);

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
  readSensors(values);
  appendSample(values);
  printSensors(values);

  if (samplesInWindow == kSeqLen) {
    runForecast();
  } else {
    Serial.printf("window: %u/%d samples\n",
                  static_cast<unsigned>(samplesInWindow), kSeqLen);
  }
}