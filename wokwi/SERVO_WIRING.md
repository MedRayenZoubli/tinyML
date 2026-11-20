# ESP32 wiring for two servo motors

This wiring uses the ESP32 for PWM control and a separate regulated servo
power supply for both motors.

| Servo A pin | Servo B pin | ESP32 / supply |
|---|---|---|
| Signal (usually orange or yellow) | - | GPIO 25 (`D25`) |
| - | Signal (usually orange or yellow) | GPIO 26 (`D26`) |
| Positive (red) | Positive (red) | 5 V servo supply positive |
| Ground (brown or black) | Ground (brown or black) | ESP32 GND and servo supply GND |

The two motor grounds must be electrically connected to the ESP32 ground.
Do not connect the servo positive wires to ESP32 `3V3`.

## Recommended connections

```text
ESP32 GPIO 25 (D25) ---- orange/yellow wire ---- Servo A signal
ESP32 GPIO 26 (D26) ---- orange/yellow wire ---- Servo B signal

ESP32 GND ----------------+----------------------- Servo A ground
                          +----------------------- Servo B ground
                          |
5-6 V servo supply GND ---+

5-6 V servo supply positive --------+---------- Servo A positive
                                    +---------- Servo B positive
```

Use a 5–6 V supply rated for the combined stall current of both servos.
Many small hobby servos draw 1 A or more each while moving. Add a 470–1000 uF
capacitor close to the servo supply terminals. If the regulator or servo current
is not sufficient, use a separate motor-grade power supply with its positive
output isolated from the ESP32 and only the grounds connected together.

Disconnect servo power while changing wiring. Keep servo signal wires away from
high-current supply wires where practical.
