#include <Arduino.h>
#include <ESP32Servo.h>

constexpr int kGateServoAPin = 25;
constexpr int kGateServoBPin = 26;

Servo gateServoA;
Servo gateServoB;
int servoAAngle = 90;
int servoBAngle = 90;

void printStatus() {
  Serial.printf(
      "servoA=%d servoB=%d\n"
      "Commands: A<angle>, B<angle>, 90, C, O, S, P, R\n",
      servoAAngle, servoBAngle);
}

void setA(int angle) {
  angle = constrain(angle, 0, 180);
  servoAAngle = angle;
  gateServoA.write(angle);
  Serial.printf("servoA=%d\n", angle);
}

void setB(int angle) {
  angle = constrain(angle, 0, 180);
  servoBAngle = angle;
  gateServoB.write(angle);
  Serial.printf("servoB=%d\n", angle);
}

bool isAngleCommand(const String& command) {
  if (command.length() == 0) {
    return false;
  }
  for (unsigned int i = 0; i < command.length(); ++i) {
    if (!isDigit(command.charAt(i))) {
      return false;
    }
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  gateServoA.attach(kGateServoAPin);
  gateServoB.attach(kGateServoBPin);
  gateServoA.write(servoAAngle);
  gateServoB.write(servoBAngle);

  Serial.println("\nESP32 servo calibration");
  Serial.println("POWER OFF before wiring. Use a separate 5-6V servo supply");
  Serial.println("with the grounds connected to ESP32 GND.");
  Serial.println("Move in small steps and stop before the linkage binds.");
  printStatus();
}

void loop() {
  if (!Serial.available()) {
    delay(20);
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();
  if (command.length() == 0) {
    return;
  }

  char selector = command.charAt(0);
  if (selector == 'A' || selector == 'B') {
    int angle = command.substring(1).toInt();
    if (angle < 0 || angle > 180) {
      Serial.println("Angle must be from 0 to 180.");
    } else if (selector == 'A') {
      setA(angle);
    } else {
      setB(angle);
    }
  } else if (command.charAt(0) == 'P') {
    printStatus();
  } else if (command == "R") {
    setA(90);
    setB(90);
  } else if (command == "C") {
    setA(10);
    setB(170);
  } else if (command == "O") {
    setA(170);
    setB(10);
  } else if (command == "S") {
    Serial.println("Sweeping both servos safely...");
    for (int angle = 10; angle <= 170; angle += 5) {
      gateServoA.write(angle);
      gateServoB.write(180 - angle);
      delay(50);
    }
    for (int angle = 170; angle >= 10; angle -= 5) {
      gateServoA.write(angle);
      gateServoB.write(180 - angle);
      delay(50);
    }
    servoAAngle = 10;
    servoBAngle = 170;
    printStatus();
  } else if (isAngleCommand(command)) {
    int angle = command.toInt();
    setA(angle);
    setB(180 - angle);
  } else {
    Serial.println("Unknown command.");
    printStatus();
  }
}
