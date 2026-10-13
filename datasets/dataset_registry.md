# Biomedical Dataset Registry & Audit Report

This registry documents all clinical and physiological benchmark datasets evaluated for the **Multimodal Health & Access System** in compliance with PhysioNet guidelines, open-science licensing, and human research protections.

---

## 1. ECG Reference Benchmark: MIT-BIH Arrhythmia Database

| Attribute | Details |
| :--- | :--- |
| **Dataset Name** | MIT-BIH Arrhythmia Database |
| **Source URL** | https://physionet.org/content/mitdb/1.0.0/ |
| **DOI / Citation** | Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine* 20(3):45-50 (May-June 2001). |
| **License** | Open Data Commons Attribution License v1.0 (ODC-By v1.0) |
| **Access Tier** | Open Access (Publicly available, no credentialing required) |
| **Signal Channels** | 2-channel ambulatory ECG (Lead II and modified lead V1, V2, V4, or V5) |
| **Sampling Rate** | 360 Hz per channel |
| **ADC Resolution** | 11-bit ADC over a $\pm 10\text{ mV}$ dynamic range (200 ADC units per mV) |
| **Ground Truth Annotations** | Expert cardiologist-annotated R-peak locations and beat classifications (Normal, Premature Ventricular Contractions, Atrial Premature Contractions, etc.) |
| **Resampling Pipeline** | Resampled to target **100 Hz** via `scipy.signal.resample` polyphase filtering to match Arduino Uno R3 ADC streaming rate |
| **Intended System Use** | Benchmarking Butterworth bandpass filtering (0.5–40 Hz), 50/60 Hz notch filtering, Pan-Tompkins R-peak detector sensitivity ($Se$) and positive predictivity ($+P$), R-R interval precision, and time-domain HRV metrics (SDNN, RMSSD, pNN50) |
| **Known Limitations** | Holter recordings contain high baseline wander, muscle movement artifacts (EMG), and PVC ectopic beats; does not include paired photoplethysmogram (PPG) or continuous arterial blood pressure |

---

## 2. PPG & Respiration Benchmark: BIDMC PPG and Respiration Dataset

| Attribute | Details |
| :--- | :--- |
| **Dataset Name** | BIDMC PPG and Respiration Dataset |
| **Source URL** | https://physionet.org/content/bidmc/1.0.0/ |
| **DOI / Citation** | Pimentel MA, Johnson AE, Charlton PH, et al. Toward a Robust Estimation of Respiratory Rate from Pulse Oximeters. *IEEE Transactions on Biomedical Engineering*, 64(8):1914-1923, 2017. |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Access Tier** | Open Access (Publicly available) |
| **Signal Channels** | Photoplethysmogram (PPG), Respiratory signal (impedance pneumography), and Lead II ECG |
| **Sampling Rate** | 125 Hz |
| **ADC Resolution** | 16-bit digitizer |
| **Ground Truth Annotations** | Manual breath-by-breath annotations, reference respiratory rate, reference heart rate, and reference SpO2 |
| **Resampling Pipeline** | Resampled from 125 Hz to **100 Hz** via polyphase decimation/interpolation (`datasets.data_loader.DatasetLoader.resample_signal`) |
| **Intended System Use** | Validation of PPG systolic pulse peak detection, pulse rate calculation, respiratory rate estimation via ECG-Derived Respiration (EDR) and Respiratory-Induced Amplitude Variations (RIAM), and SpO2 ratio-of-ratios algorithm |
| **Known Limitations** | Primary public CSV records contain single infrared PPG channel; dual-wavelength Red (660nm) and IR (880nm) photodiode calibration requires controlled optical phantom or paired multi-wavelength transmission data |

---

## 3. Synchronized ECG, PPG & Arterial Pressure: MIMIC-III Waveform Matched Subset

| Attribute | Details |
| :--- | :--- |
| **Dataset Name** | MIMIC-III Waveform Database Matched Subset |
| **Source URL** | https://physionet.org/content/mimic3wdb-matched/1.0/ |
| **DOI / Citation** | Johnson AEW, Pollard TJ, Shen L, Lehman LH, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, Mark RG. MIMIC-III, a freely accessible critical care database. *Scientific Data* 3:160035 (2016). |
| **License** | PhysioNet Credentialed Health Data License 1.5.0 |
| **Access Tier** | **Credentialed Access**: Requires CITI Data or Specimens Only Research certification, human subjects compliance, signed Data Use Agreement (DUA), and authenticated PhysioNet account |
| **Signal Channels** | Continuous synchronized Lead II ECG, Fingertip Photoplethysmogram (PLETH), and Invasive Arterial Blood Pressure (ABP) via indwelling arterial catheter |
| **Sampling Rate** | 125 Hz |
| **Reference Labels** | Beat-to-beat Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) from arterial lines |
| **Intended System Use** | Training, feature extraction, and benchmark regression of cuffless blood pressure models using Pulse Arrival Time (PAT) and PPG pulse morphometry |
| **PhysioNet Compliance Note** | **Strictly Prohibited from Git Repository**: In compliance with the PhysioNet Data Use Agreement, raw MIMIC-III waveforms and derived clinical participant data must **never** be committed to public or uncredentialed git repositories. |

---

## 4. Synthetic Mock & Fixture Framework for CI/CD

To ensure fast, reliable, reproducible unit testing without requiring gigabyte-scale medical database downloads or violating credentialed data licensing:

- The test suite utilizes programmatic synthetic fixtures implemented in [`datasets/data_loader.py`](file:///d:/PROJECTS/ESA/datasets/data_loader.py):
  1. `generate_synthetic_ecg_fixture(duration_s, fs, heart_rate, noise_std)`: Synthesizes physiologically accurate P-Q-R-S-T complexes with known millisecond R-peak ground truth locations.
  2. `generate_synthetic_ppg_fixture(duration_s, fs, heart_rate, pat_delay_ms)`: Synthesizes paired Red and Infrared photoplethysmograms with realistic systolic peaks, dicrotic notches, and configurable Pulse Arrival Time (PAT) delays.
- Used across:
  - [`tests/test_ecg.py`](file:///d:/PROJECTS/ESA/tests/test_ecg.py)
  - [`tests/test_ppg.py`](file:///d:/PROJECTS/ESA/tests/test_ppg.py)
  - [`tests/test_combined.py`](file:///d:/PROJECTS/ESA/tests/test_combined.py)
  - [`tests/test_ml_bp.py`](file:///d:/PROJECTS/ESA/tests/test_ml_bp.py)
- All 20 tests pass deterministically across Windows and macOS environments in under 15 seconds.
