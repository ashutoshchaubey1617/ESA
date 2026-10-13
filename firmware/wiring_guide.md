# Embedded Hardware Wiring Guide & Safety Precautions

Comprehensive technical guide for interfacing the **Analog Devices AD8232** single-lead ECG front-end and the **Maxim Integrated MAX30102** dual-wavelength optical PPG sensor with an **Arduino Uno R3** microcontroller.

---

## 1. Hardware Specifications & Power Budgets

| Component | Part / IC | Logic / Rail Voltage | Operating Voltage | Current Consumption | Interface |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Microcontroller** | Arduino Uno R3 (ATmega328P) | 5.0 V Logic | 5.0 V (USB / Barrel Jack) | ~45 mA base | USB Serial (115200 Baud) |
| **PPG Sensor** | Maxim MAX30102 | 1.8 V Core / 3.3 V LED | 3.3 V Breakout VCC | 0.6 mA - 1.2 mA (peak) | I2C (Address: `0x57`, 400 kHz) |
| **ECG Front-End**| Analog Devices AD8232 | 3.3 V Analog | 2.0 V - 3.5 V (Nominal 3.3 V) | 170 µA (ultra-low power) | Analog OUT (A0) + Dig. Leads-off |

---

## 2. Voltage Safety & Electrical Precautions

> [!CAUTION]
> **CRITICAL VOLTAGE HAZARD — READ BEFORE CONNECTING POWER**:
> - **MAX30102**: The MAX30102 sensor IC requires 1.8V for its digital core and 3.3V for internal LED drivers. While many breakout modules integrate a 3.3V LDO, connecting the VIN/VCC pin to the Arduino's **5.0V pin** will overload the sensor or expose the internal die to overvoltage, causing **irreversible hardware destruction**.
> - **AD8232**: The AD8232 instrumentation amplifier has an **absolute maximum rating of 3.6V**. Powering it from the Arduino's 5.0V pin will burn out internal bias generators and operational amplifiers.
> - **Arduino 3.3V Rail Current Limit**: The Arduino Uno R3 provides regulated 3.3V power via an on-board LP2985 LDO regulator rated for a maximum of 50 mA. The combined draw of the MAX30102 (~1.2 mA) and AD8232 (~0.17 mA) is approximately 1.4 mA, well within safe thermal limits. **Never power external high-drain devices (e.g. motors, relays, heaters) from this 3.3V rail.**

---

## 3. Circuit Connection Schematic

```
                                ARDUINO UNO R3
                           +----------------------+
                           |                      |
                           |                 3.3V +--------+--------------------+
                           |                      |        |                    |
                           |                  GND +----+   | (3.3V Rail)        | (3.3V Rail)
                           |                      |    |   |                    |
                           |                   A0 +<---|---|------------+       |
                           |                      |    |   |            |       |
                           |                   A4 +<---|---|----+       |       |
                           |                 (SDA)|    |   |    |       |       |
                           |                   A5 +<---|---|--+ |       |       |
                           |                 (SCL)|    |   |  | |       |       |
                           |                      |    |   |  | |       |       |
                           |                  D10 +<---|---|--|-|-------|-+     |
                           |                 (LO+)|    |   |  | |       | |     |
                           |                  D11 +<---|---|--|-|-------|-|-+   |
                           |                 (LO-)|    |   |  | |       | | |   |
                           +----------------------+    |   |  | |       | | |   |
                                                       |   |  | |       | | |   |
              +----------------------------------------+   |  | |       | | |   |
              |                                            |  | |       | | |   |
              |                                            |  | |       | | |   |
              v                                            v  v v       | | |   |
       +--------------+                               +--------------+  | | |   |
       |     GND      |                               |     VIN      |  | | |   |
       |              |                               |              |  | | |   |
       |   MAX30102   |                               |   MAX30102   |  | | |   |
       |  PPG Sensor  |                               |  PPG Sensor  |  | | |   |
       |              |                               |  SCL    SDA  |  | | |   |
       +--------------+                               +---+------+---+  | | |   |
                                                          |      |      | | |   |
                                                          +------+      | | |   |
                                                                        | | |   |
                                                                        v v v   v
                                                                   +--------------+
                                                                   | OUTPUT   3.3V|
                                                                   |              |
                                                                   |    AD8232    |
                                                                   |  ECG Sensor  |
                                                                   |              |
                                                                   | GND  LO+ LO- |
                                                                   +--+----+---+--+
                                                                      |    |   |
                                                                      v    v   v
                                                                     GND  D10 D11
```

---

## 4. Pin Interconnections & Wiring Table

### A. MAX30102 Pulse Oximeter Module (I2C)
| MAX30102 Breakout Pin | Arduino Uno R3 Pin | Wire Color (Recommended) | Description & Electrical Function |
| :--- | :--- | :--- | :--- |
| **VIN / VCC** | **3.3V** | **Red** | Regulated 3.3V power supply (**DO NOT USE 5V**) |
| **GND** | **GND** | **Black** | System common ground |
| **SDA** | **A4** (or dedicated SDA) | **Blue / Green** | I2C Data line (Fast-Mode 400 kHz) |
| **SCL** | **A5** (or dedicated SCL) | **Yellow** | I2C Clock line |
| **INT** | *Not connected* (or D2) | Gray | Interrupt line (firmware uses non-blocking polling) |

### B. AD8232 Single-Lead ECG Front-End (Analog + Digital)
| AD8232 Breakout Pin | Arduino Uno R3 Pin | Wire Color (Recommended) | Description & Electrical Function |
| :--- | :--- | :--- | :--- |
| **3.3V** | **3.3V** | **Red** | Power rail (operating range 2.0V - 3.5V) |
| **GND** | **GND** | **Black** | System common ground |
| **OUTPUT** | **A0** | **Orange** | Filtered analog ECG signal (0.0 V - 3.3 V range, centered at ~1.65 V) |
| **LO+** | **D10** | **Purple** | Lead-off detection positive electrode (Active HIGH) |
| **LO-** | **D11** | **Brown** | Lead-off detection negative electrode (Active HIGH) |
| **SDN** | *Not connected* | White | Shutdown control (Active LOW; pull to 3.3V or leave open for normal run) |

---

## 5. Breadboard Assembly Step-by-Step

1. **Power Isolation**: Disconnect the Arduino Uno USB cable from your host computer before wiring.
2. **Ground & Power Rails**:
   - Run a jumper wire from Arduino **GND** to the breadboard negative blue rail.
   - Run a jumper wire from Arduino **3.3V** to the breadboard positive red rail.
   - *Never connect Arduino 5V to the breadboard power rail.*
3. **MAX30102 Wiring**:
   - Connect MAX30102 `GND` to the breadboard negative rail.
   - Connect MAX30102 `VIN` to the breadboard 3.3V rail.
   - Connect MAX30102 `SDA` to Arduino `A4`.
   - Connect MAX30102 `SCL` to Arduino `A5`.
4. **AD8232 Wiring**:
   - Connect AD8232 `GND` to the breadboard negative rail.
   - Connect AD8232 `3.3V` to the breadboard 3.3V rail.
   - Connect AD8232 `OUTPUT` to Arduino `A0`.
   - Connect AD8232 `LO+` to Arduino `D10`.
   - Connect AD8232 `LO-` to Arduino `D11`.
5. **Physical Inspection**: Double-check all power connections before inserting the USB cable.

---

## 6. Electrode Placement (Einthoven's Lead II Configuration)

The system records a standard Lead II vector (right shoulder to left hip), yielding the highest typical R-peak amplitude for robust Pan-Tompkins QRS detection.

```
                  (Right)                     (Left)
                     RA                         LA
                     [o]                       [o]
                       \                       /
                        \       (Heart)       /
                         \        <3         /
                          \                 /
                           \               /
                            \             /
                             \           /
                              v         v
                                        RL (Reference Ground)
                                        [o]
```

### Placement Coordinates:
1. **RA (Right Arm / Red Cable)**: Right infraclavicular fossa (just below the right collarbone) or inner right forearm.
2. **LA (Left Arm / Yellow Cable)**: Left infraclavicular fossa (just below the left collarbone) or inner left forearm.
3. **RL (Right Leg / Green Cable)**: Lower right abdomen or right lower wrist. Acts as the Driven Right Leg (DRL) reference to eliminate 50/60 Hz mains interference.

### Skin Preparation Guidelines:
- Clean contact skin with an alcohol swab to remove epidermal oils and dead cells.
- Allow skin to dry completely before applying fresh Ag/AgCl disposable hydrogel electrodes.
- Secure electrode leads to reduce motion artifacts from pulling cables.

---

## 7. Firmware Uploading & Serial Validation

### Firmware Compilation & Flashing:
1. Launch the **Arduino IDE** (version 2.0+ recommended).
2. Install the **SparkFun MAX3010x Sensor Library** via Library Manager (`Tools > Manage Libraries... > search "SparkFun MAX3010x"`).
3. Open [`firmware/arduino_health_monitor.ino`](file:///d:/PROJECTS/ESA/firmware/arduino_health_monitor.ino).
4. Connect the Arduino Uno to your PC/Mac via USB.
5. In Arduino IDE:
   - Select Board: **Tools > Board > Arduino AVR Boards > Arduino Uno**.
   - Select Port: **Tools > Port > COMx** (Windows) or `/dev/cu.usbmodem*` (macOS).
6. Click **Upload** (Ctrl+U / Cmd+U).

### Serial Packet Protocol (100 Hz Streaming):
The Arduino streams ASCII comma-separated values at **115200 baud**:

```text
SAMPLE,<timestamp_ms>,<seq_number>,<ecg_raw>,<ppg_red>,<ppg_ir>,<leads_off>
```

| Field | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `SAMPLE` | String | Fixed literal | Packet start identifier |
| `timestamp_ms` | Unsigned Long | $0 - 4294967295$ | Milliseconds since microcontroller boot (`millis()`) |
| `seq_number` | Unsigned Long | $0 - 4294967295$ | Monotonically incrementing packet counter for loss detection |
| `ecg_raw` | Integer | $0 - 1023$ | 10-bit analog conversion of AD8232 OUTPUT on A0 |
| `ppg_red` | Unsigned Long | $0 - 262143$ | 18-bit optical intensity (Red 660 nm) |
| `ppg_ir` | Unsigned Long | $0 - 262143$ | 18-bit optical intensity (Infrared 880 nm) |
| `leads_off` | Integer | $0$ or $1$ | $1$ if either `LO+` or `LO-` is triggered (lead detached) |

Example Raw Stream:
```text
SAMPLE,14520,1452,512,65420,68100,0
SAMPLE,14530,1453,514,65415,68095,0
SAMPLE,14540,1454,580,65430,68120,0
```

---

## 8. Hardware Troubleshooting Matrix

| Symptom | Probable Cause | Diagnostic & Corrective Action |
| :--- | :--- | :--- |
| **`leads_off` = 1 continuously** | Loose electrode or unseated D10/D11 wire | 1. Check if electrodes have dried out; replace pads.<br>2. Verify D10 connects to LO+ and D11 connects to LO-.<br>3. Inspect common GND connection. |
| **PPG Red/IR stays near 0 or tissue contact false** | Finger not properly covering optical cavity | 1. Rest finger gently on MAX30102 sensor glass without excessive pressure.<br>2. Verify I2C wiring (A4 = SDA, A5 = SCL).<br>3. Run I2C scanner sketch to confirm device is responsive at `0x57`. |
| **Severe 50/60 Hz noise on ECG** | Missing reference electrode or ungrounded laptop charger | 1. Ensure green RL electrode is firmly attached to skin.<br>2. Disconnect noisy AC laptop switch-mode power supplies; run on battery power to isolate common-mode noise.<br>3. Confirm software 50 Hz notch filter is enabled. |
| **Serial port access denied** | Port held open by Arduino Serial Monitor or another app | Close Arduino IDE Serial Monitor or Serial Plotter before launching the Streamlit dashboard or tests. |
| **MAX30102 IC gets warm** | Incorrect power rail (5V connected instead of 3.3V) | **IMMEDIATELY UNPLUG USB.** Verify wiring. 5V will destroy the optical sensor IC. |
