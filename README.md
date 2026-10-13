# Multimodal Health & Access Monitoring System

An integrated clinical and biometric edge computing system combining:
1. **Biometric Face Verification & Tracking**: Real-time SCRFD face detection, 5-point landmark alignment (`norm_crop`), ArcFace iResNet-100 embeddings, multi-target BYTETracker, and anti-spoofing micro-motion liveness verification.
2. **Embedded Physiological Sensing**: Arduino Uno R3 streaming dual-channel MAX30102 PPG (Red & Infrared) and AD8232 single-lead ECG at 100 Hz over USB serial.
3. **Biomedical Signal Processing**: Digital Butterworth & notch filtering, Pan-Tompkins R-peak detection, R-R intervals, Heart Rate, HRV (RMSSD, SDNN, pNN50), Ratio-of-Ratios SpO2, Pulse Arrival Time (PAT), and Respiratory Rate (EDR).
4. **Machine Learning Analytics**: Cuffless Blood Pressure regression pipeline (Random Forest) based on synchronized PAT and PPG pulse morphometry.
5. **Unified Dashboard & SQLite Persistence**: Multi-view Streamlit dashboard with session management, participant enrollment, real-time oscilloscopes, technical alert banners, and CSV export.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Laptop Camera / Webcam    |
                      +--------------+--------------+
                                     |
                                     v
                 +---------------------------------------+
                 |       Face Recognition Service        |
                 | - SCRFD 2.5G ONNX Face Detector       |
                 | - 5-Point Landmark norm_crop          |
                 | - ArcFace iResNet-100 Embeddings      |
                 | - BYTETracker Multi-target Tracking   |
                 | - Landmark Dynamics Liveness Detector |
                 +-------------------+-------------------+
                                     |
                                     v
+------------------+    +----------------------------+    +-----------------------+
|  Arduino Uno R3  |--->|    SerialManager (Host)    |--->|  Biomedical Engine    |
| - MAX30102 (I2C) |    | - Non-blocking USB Serial  |    | - ECG Filter & Pan-T. |
| - AD8232 (A0/LO) |    | - Auto-reconnect & queue   |    | - PPG & Ratio SpO2    |
| - 100 Hz Timer   |    | - Synthetic Simulator Mode |    | - PAT & Respiration   |
+------------------+    +----------------------------+    +-----------+-----------+
                                                                      |
                                                                      v
                                                   +------------------------------------+
                                                   |      Unified Streamlit App         |
                                                   | - 8 Clinical & Diagnostic Views    |
                                                   | - SQLite Participant & Session DB  |
                                                   | - Raw Telemetry & Summary Exporter |
                                                   | - Random Forest BP Model Pipeline  |
                                                   +------------------------------------+
```

---

## Hardware Connection Schematics & Safety

### Connection Schematic

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
              v                                            v  v v       | | |   |
       +--------------+                               +--------------+  | | |   |
       |  MAX30102    |                               |  MAX30102    |  | | |   |
       |  PPG Sensor  |                               |  PPG Sensor  |  | | |   |
       |     GND      |                               |  VIN SCL SDA |  | | |   |
       +--------------+                               +---+---+---+--+  | | |   |
                                                          |   |   |     | | |   |
                                                          +---+---+     | | |   |
                                                                        | | |   |
                                                                        v v v   v
                                                                   +--------------+
                                                                   | OUTPUT   3.3V|
                                                                   |    AD8232    |
                                                                   |  ECG Sensor  |
                                                                   | GND  LO+ LO- |
                                                                   +--+----+---+--+
                                                                      |    |   |
                                                                      v    v   v
                                                                     GND  D10 D11
```

### Pin Interconnections
| Sensor Module | Sensor Pin | Arduino Uno R3 Pin | Function / Description |
| :--- | :--- | :--- | :--- |
| **MAX30102** | VIN | **3.3V** | Power rail (**DO NOT connect to 5V**) |
| **MAX30102** | GND | **GND** | System common ground |
| **MAX30102** | SDA | **A4** (or dedicated SDA)| I2C Data line (Fast Mode 400kHz) |
| **MAX30102** | SCL | **A5** (or dedicated SCL)| I2C Clock line |
| **AD8232** | 3.3V | **3.3V** | Power rail (**Absolute Max Rating: 3.6V**) |
| **AD8232** | GND | **GND** | System common ground |
| **AD8232** | OUTPUT | **A0** | Analog ECG output (0.0 V - 3.3 V range, centered at ~1.65 V) |
| **AD8232** | LO+ | **D10** | Lead-off Detection Positive (Digital input) |
| **AD8232** | LO- | **D11** | Lead-off Detection Negative (Digital input) |

> [!CAUTION]
> **CRITICAL VOLTAGE HAZARD**:
> - Never connect the `VCC`/`VIN` pins of either the **MAX30102** or **AD8232** to the Arduino's **5.0V** pin. Both ICs are 3.3V maximum devices. 5V will cause irreversible physical burnout.
> - The Arduino Uno R3 on-board 3.3V regulator (LP2985) has a 50 mA limit. The combined sensor current draw is ~1.4 mA, well within limits. Do not add high-current peripherals to the 3.3V rail.
> - For full schematic details, breadboard wiring guide, and electrode placement, refer to [`firmware/wiring_guide.md`](file:///d:/PROJECTS/ESA/firmware/wiring_guide.md).

### Firmware Upload
1. Open the Arduino IDE.
2. Install the **SparkFun MAX3010x Sensor Library** (`Tools > Manage Libraries...`).
3. Open [`firmware/arduino_health_monitor.ino`](file:///d:/PROJECTS/ESA/firmware/arduino_health_monitor.ino).
4. Connect the Arduino Uno R3 via USB.
5. Select board **Arduino Uno** and your active serial port under **Tools > Port**.
6. Click **Upload** (Ctrl+U / Cmd+U).

---

## Software Installation & Setup

### Windows Setup (PowerShell)

```powershell
# 1. Open PowerShell and navigate to the project directory
cd d:\PROJECTS\ESA

# 2. Allow local script execution (if virtualenv activation is restricted)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip and install all project dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Check connected COM ports (optional, for physical Arduino)
Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Description
```

### macOS Setup (Terminal)

```bash
# 1. Open Terminal and navigate to the project directory
cd /path/to/ESA

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install all project dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Check connected USB serial ports (optional, for physical Arduino)
ls /dev/cu.usb*
```

> [!NOTE]
> **macOS Camera Permission**: When running the face recognition service, macOS requires camera authorization. If prompted, allow Terminal or iTerm to access the camera in **System Settings > Privacy & Security > Camera**.

---

## Model Weights Setup

The biometric pipeline uses pre-trained weights for high-accuracy face detection and 512-dimensional feature embedding:
- **SCRFD 2.5G ONNX**: Target path [`face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx`](file:///d:/PROJECTS/ESA/face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx)
- **ArcFace iResNet-100**: Target path [`face_recognition/arcface/weights/arcface_r100.pth`](file:///d:/PROJECTS/ESA/face_recognition/arcface/weights/arcface_r100.pth)

### Automatic Download Command

Run the following command from your active virtual environment:

#### Windows (PowerShell):
```powershell
# Download SCRFD detector weights
python -c "import urllib.request, os; os.makedirs('face_detection/scrfd/weights', exist_ok=True); urllib.request.urlretrieve('https://huggingface.co/MonsterMMORPG/files1/resolve/main/scrfd_2.5g_bnkps.onnx', 'face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx')"

# Download ArcFace recognition weights
python -c "import urllib.request, os; os.makedirs('face_recognition/arcface/weights', exist_ok=True); urllib.request.urlretrieve('https://huggingface.co/BooBooWu/Vec2Face/resolve/main/weights/arcface-r100-glint360k.pth', 'face_recognition/arcface/weights/arcface_r100.pth')"
```

#### macOS / Linux (Terminal):
```bash
# Download SCRFD detector weights
python3 -c "import urllib.request, os; os.makedirs('face_detection/scrfd/weights', exist_ok=True); urllib.request.urlretrieve('https://huggingface.co/MonsterMMORPG/files1/resolve/main/scrfd_2.5g_bnkps.onnx', 'face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx')"

# Download ArcFace recognition weights
python3 -c "import urllib.request, os; os.makedirs('face_recognition/arcface/weights', exist_ok=True); urllib.request.urlretrieve('https://huggingface.co/BooBooWu/Vec2Face/resolve/main/weights/arcface-r100-glint360k.pth', 'face_recognition/arcface/weights/arcface_r100.pth')"
```

### Manual Mirrors (Google Drive):
- SCRFD weights mirror: [Google Drive link](https://drive.google.com/drive/folders/1C9RzReAihJQRl8EJOX6vQj7qbHBPmzME?usp=sharing)
- ArcFace weights mirror: [Google Drive link](https://drive.google.com/drive/folders/1CHHb_7wbvfjKPFNKVBb76lL5sVfBLcv5?usp=sharing)

---

## Launching the Unified Application

Launch the Streamlit web dashboard with a single command:

```shell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Hardware Simulation Mode
If you do not have physical Arduino hardware connected, check **"Simulate Hardware (Synthetic Stream)"** in the sidebar or under the **Settings** view. The built-in generator streams realistic 100 Hz synthetic Lead II ECG, dual-wavelength PPG, and synchronized cardiac beats for full end-to-end evaluation.

### Dashboard Navigation:
1. **Overview Dashboard**: System-wide status, live heart rate, SpO2, PAT, HRV, respiration, and technical alert banners.
2. **Face Recognition & Access**: Live camera preview, participant enrollment, cosine similarity matching, landmark dynamics liveness verification, and session binding.
3. **Live ECG**: Real-time Lead II ECG oscilloscope, filtered vs raw ADC toggle, detected R-peaks, and R-R interval chart.
4. **Live PPG & SpO2**: Dual Red (660nm) and Infrared (880nm) photoplethysmograms, systolic peak markers, and tissue contact indicator.
5. **Health Analytics**: ECG vs PPG concordance check, Pulse Arrival Time (PAT) breakdown, and cuffless blood pressure analysis.
6. **Session History**: Participant-filtered historical session directory with time-domain metric charts and session deletion.
7. **Reports & Export**: Download timestamped raw 100 Hz sensor telemetry and summary metrics in CSV format.
8. **Settings**: Serial port selection, baud rate configuration, filter parameter inspection, and hardware simulation toggle.

---

## Running the Automated Test Suite

Execute the comprehensive test suite with `pytest`:

```shell
pytest -v
```

All 28 test cases cover:
- Bandpass filtering, notch filtering, and Pan-Tompkins R-peak detection.
- R-R intervals, heart rate, and time-domain HRV (RMSSD, SDNN, pNN50).
- PPG filtering, systolic peak detection, tissue contact detection, and SpO2 ratio-of-ratios.
- Pulse Arrival Time (PAT) synchronization and respiration rate estimation.
- Serial packet framing, checksum validation, buffer safety, and synthetic hardware simulation.
- Participant enrollment, session persistence, cascading schema integrity, and SQLite database CRUD.
- Random Forest blood pressure training and clinical safety gating.
- Face alignment (`norm_crop`), ArcFace embeddings, and anti-spoofing liveness.

---

## Scientific Definitions & Disclaimers

### Pulse Arrival Time (PAT) vs Pulse Transit Time (PTT)
- **Pulse Arrival Time (PAT)**: The latency between ventricular electrical depolarization (ECG R-peak) and the arrival of the pulse pressure wave at the peripheral measurement site (PPG systolic peak).
  $$\text{PAT} = \text{PEP} + \text{PTT}$$
- **Pre-Ejection Period (PEP)**: The electromechanical delay and isovolumetric contraction time of the left ventricle before aortic valve opening.
- **Pulse Transit Time (PTT)**: The true propagation time through the arterial tree.
Because non-invasive surface sensors cannot detect the microsecond instant of aortic valve opening directly, our system measures and reports PAT.

### Clinical Disclaimer
> [!IMPORTANT]
> This application is intended for research, educational, and wellness monitoring purposes only. It is not an FDA/CE-cleared diagnostic medical device. Cuffless blood pressure estimation is research-grade and requires individual cuff calibration. Do not use for emergency or acute diagnostic decisions.
