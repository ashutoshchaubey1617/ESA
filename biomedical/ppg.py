"""
MAX30102 PPG and Pulse-Oximetry Signal Processing Module
Implements bandpass filtering, pulse peak detection, PPG-derived heart rate,
SpO2 estimation via Ratio-of-Ratios, and finger contact detection.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal


class PPGProcessor:
    """Robust PPG and SpO2 processing engine for MAX30102 Red and IR optical signals."""

    def __init__(self, sampling_rate: float = 100.0, min_ir_dc_threshold: float = 25000.0):
        self.fs = float(sampling_rate)
        self.min_ir_dc_threshold = min_ir_dc_threshold

    def check_finger_contact(self, raw_red: np.ndarray, raw_ir: np.ndarray) -> Dict[str, Any]:
        """
        Determines whether a finger is properly placed on the sensor glass.
        MAX30102 returns low counts (< 25,000) when unobstructed, and > 50,000 when illuminated tissue is present.
        """
        if raw_ir is None or raw_red is None or len(raw_ir) == 0 or len(raw_red) == 0:
            return {"contact": False, "reason": "No data", "ir_dc": 0.0, "red_dc": 0.0}

        ir_dc = float(np.mean(raw_ir))
        red_dc = float(np.mean(raw_red))

        # Check for sensor saturation / ambient optical overdrive (18-bit ADC: max 262143)
        if ir_dc >= 255000.0 or red_dc >= 255000.0:
            return {
                "contact": False,
                "reason": "Sensor saturated / optical overdrive",
                "ir_dc": round(ir_dc, 1),
                "red_dc": round(red_dc, 1),
            }

        if ir_dc < self.min_ir_dc_threshold:
            return {
                "contact": False,
                "reason": "Finger not detected on sensor (low optical return)",
                "ir_dc": round(ir_dc, 1),
                "red_dc": round(red_dc, 1),
            }

        return {
            "contact": True,
            "reason": "Optimal tissue contact",
            "ir_dc": round(ir_dc, 1),
            "red_dc": round(red_dc, 1),
        }

    def filter_ppg(
        self,
        raw_signal: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 5.0,
    ) -> np.ndarray:
        """
        Bandpass filters PPG waveform (0.5 - 5.0 Hz = 30 to 300 BPM)
        to isolate the AC cardiac pulsatile component from DC baseline and high-frequency noise.
        """
        if raw_signal is None or len(raw_signal) == 0:
            return np.array([], dtype=np.float64)

        data = np.asarray(raw_signal, dtype=np.float64)
        if len(data) < int(self.fs * 0.5):
            return data - np.mean(data) if len(data) > 0 else data

        # Remove DC baseline
        data = data - np.median(data)

        nyq = 0.5 * self.fs
        low = max(0.01, min(lowcut / nyq, 0.99))
        high = max(low + 0.01, min(highcut / nyq, 0.99))

        b, a = signal.butter(2, [low, high], btype="bandpass")
        filtered = signal.filtfilt(b, a, data)
        return filtered

    def detect_pulse_peaks(
        self, filtered_ppg: np.ndarray, min_distance_ms: float = 300.0
    ) -> np.ndarray:
        """
        Detects systolic pulse peaks from the AC pulsatile PPG signal.
        """
        if filtered_ppg is None:
            return np.array([], dtype=int)

        data = np.asarray(filtered_ppg, dtype=np.float64)
        if len(data) < int(self.fs):
            return np.array([], dtype=int)

        # Flatline / negligible signal check
        if np.ptp(data) < 1e-3 or np.std(data) < 1e-3:
            return np.array([], dtype=int)

        # Systolic peaks are prominent positive excursions
        threshold = np.mean(data) + 0.3 * np.std(data)
        min_distance = max(1, int((min_distance_ms / 1000.0) * self.fs))

        peaks, _ = signal.find_peaks(data, height=threshold, distance=min_distance)
        return peaks

    def calculate_heart_rate(self, peak_indices: Optional[np.ndarray]) -> Optional[float]:
        """Calculates PPG-derived Heart Rate (Pulse Rate in BPM)."""
        if peak_indices is None or len(peak_indices) < 2:
            return None

        peak_intervals_ms = (np.diff(peak_indices) / self.fs) * 1000.0
        # Filter physiological bounds [35 - 220 BPM] -> [272 - 1714 ms]
        valid_intervals = peak_intervals_ms[
            (peak_intervals_ms >= 272.0) & (peak_intervals_ms <= 1714.0)
        ]
        if len(valid_intervals) == 0:
            return None

        mean_interval_s = np.mean(valid_intervals) / 1000.0
        hr = 60.0 / mean_interval_s
        return float(np.round(hr, 1))

    def estimate_spo2(
        self,
        raw_red: np.ndarray,
        raw_ir: np.ndarray,
        filtered_red: np.ndarray,
        filtered_ir: np.ndarray,
    ) -> Optional[float]:
        """
        Estimates arterial oxygen saturation (SpO2) using the standard Ratio-of-Ratios (R) method:
        R = (AC_red / DC_red) / (AC_ir / DC_ir)
        SpO2 = 110 - 25 * R
        """
        if (
            raw_red is None
            or raw_ir is None
            or filtered_red is None
            or filtered_ir is None
            or len(raw_red) < int(self.fs)
            or len(raw_ir) < int(self.fs)
        ):
            return None

        dc_red = float(np.mean(raw_red))
        dc_ir = float(np.mean(raw_ir))

        if dc_red <= 0 or dc_ir <= 0:
            return None

        # Guard against optical saturation
        if dc_red >= 255000.0 or dc_ir >= 255000.0:
            return None

        # AC amplitude via root-mean-square or standard deviation of the pulsatile component
        ac_red = float(np.std(filtered_red))
        ac_ir = float(np.std(filtered_ir))

        # Require genuine pulsatile AC signal in both channels (prevent flatline / missing pulse)
        if ac_ir < 1.0 or ac_red < 1.0:
            return None

        # Ratio of ratios
        r_ratio = (ac_red / dc_red) / (ac_ir / dc_ir)

        # Clinically plausible R ratio in human tissue: typically 0.25 <= R <= 1.8
        if r_ratio < 0.25 or r_ratio > 2.0:
            return None

        # Standard calibration curve
        # SpO2 = 110 - 25 * R
        spo2 = 110.0 - (25.0 * r_ratio)

        # Constrain to clinically plausible limits or return None if noisy/implausible
        if 70.0 <= spo2 <= 100.0:
            return float(np.round(spo2, 1))
        elif spo2 > 100.0 and r_ratio >= 0.25:
            return 99.0
        else:
            return None

    def process(self, raw_red: np.ndarray, raw_ir: np.ndarray) -> Dict[str, Any]:
        """Runs complete PPG processing pipeline."""
        contact_info = self.check_finger_contact(raw_red, raw_ir)
        if not contact_info["contact"]:
            return {
                "filtered_red": np.zeros_like(raw_red),
                "filtered_ir": np.zeros_like(raw_ir),
                "pulse_peaks": np.array([], dtype=int),
                "heart_rate": None,
                "spo2": None,
                "contact_status": contact_info,
            }

        filtered_red = self.filter_ppg(raw_red)
        filtered_ir = self.filter_ppg(raw_ir)

        # Use IR for peak detection (IR has significantly higher SNR in human tissue)
        peaks = self.detect_pulse_peaks(filtered_ir)
        hr = self.calculate_heart_rate(peaks)
        spo2 = self.estimate_spo2(raw_red, raw_ir, filtered_red, filtered_ir)

        return {
            "filtered_red": filtered_red,
            "filtered_ir": filtered_ir,
            "pulse_peaks": peaks,
            "heart_rate": hr,
            "spo2": spo2,
            "contact_status": contact_info,
        }
