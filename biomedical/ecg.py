"""
ECG Signal Processing Module for Multimodal Health System
Implements digital filtering, R-peak detection, R-R intervals, Heart Rate, HRV, and Signal Quality Assessment.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal


class ECGProcessor:
    """Robust ECG processing pipeline designed for AD8232 analog inputs."""

    def __init__(self, sampling_rate: float = 100.0):
        self.fs = float(sampling_rate)

    def filter_signal(
        self,
        raw_ecg: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        notch_freq: Optional[float] = 50.0,
    ) -> np.ndarray:
        """
        Applies a bandpass Butterworth filter and optional 50/60 Hz notch filter to the raw ECG.
        """
        data = np.asarray(raw_ecg, dtype=np.float64)
        if len(data) < int(self.fs * 0.5):
            return data

        # Remove DC offset / baseline
        data = data - np.median(data)

        # 4th-order Bandpass Butterworth filter
        nyq = 0.5 * self.fs
        low = max(0.01, min(lowcut / nyq, 0.99))
        high = max(low + 0.01, min(highcut / nyq, 0.99))

        b, a = signal.butter(4, [low, high], btype="bandpass")
        # Use filtfilt for zero-phase distortion
        filtered = signal.filtfilt(b, a, data)

        # Optional notch filter for powerline hum
        if notch_freq is not None and notch_freq < nyq:
            b_notch, a_notch = signal.iirnotch(notch_freq, Q=30.0, fs=self.fs)
            filtered = signal.filtfilt(b_notch, a_notch, filtered)

        return filtered

    def detect_r_peaks(
        self, filtered_ecg: Optional[np.ndarray], min_distance_ms: float = 300.0
    ) -> np.ndarray:
        """
        Detects R-peaks using derivative, squaring, and moving-average integration (Pan-Tompkins heuristic).
        Returns indices of detected R-peaks.
        """
        if filtered_ecg is None:
            return np.array([], dtype=int)

        data = np.asarray(filtered_ecg, dtype=np.float64)
        n = len(data)
        if n < int(self.fs):
            return np.array([], dtype=int)

        # Flatline / negligible signal check
        if np.ptp(data) < 1e-4 or np.std(data) < 1e-4:
            return np.array([], dtype=int)

        # 1. Five-point derivative
        diff = np.diff(data)
        diff = np.pad(diff, (0, 1), mode="edge")

        # 2. Squaring
        squared = diff**2

        # 3. Moving-window integration (~120ms window)
        win_size = max(1, int(0.12 * self.fs))
        kernel = np.ones(win_size) / win_size
        integrated = np.convolve(squared, kernel, mode="same")

        # 4. Adaptive thresholding
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        min_distance = max(1, int((min_distance_ms / 1000.0) * self.fs))

        peaks, _ = signal.find_peaks(
            integrated,
            height=threshold,
            distance=min_distance,
        )

        # 5. Fine-tune peak location to the local maximum in original filtered ECG
        refined_peaks = []
        search_radius = max(1, int(0.08 * self.fs))  # 80ms window
        for p in peaks:
            start_idx = max(0, p - search_radius)
            end_idx = min(n, p + search_radius + 1)
            local_max = start_idx + np.argmax(data[start_idx:end_idx])
            refined_peaks.append(local_max)

        # Remove duplicates
        refined_peaks = np.unique(refined_peaks)
        return refined_peaks

    def calculate_rr_intervals(self, r_peaks: Optional[np.ndarray]) -> np.ndarray:
        """Calculates R-R intervals in milliseconds."""
        if r_peaks is None or len(r_peaks) < 2:
            return np.array([], dtype=np.float64)
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / self.fs) * 1000.0
        return rr_ms

    def calculate_heart_rate(self, rr_intervals_ms: Optional[np.ndarray]) -> Optional[float]:
        """
        Calculates instantaneous / mean Heart Rate in BPM from valid RR intervals.
        Filters out non-physiological values outside [35, 220] BPM.
        """
        if rr_intervals_ms is None or len(rr_intervals_ms) == 0:
            return None

        # Physiological filter (272ms = 220 BPM, 1714ms = 35 BPM)
        valid_rr = rr_intervals_ms[(rr_intervals_ms >= 272.0) & (rr_intervals_ms <= 1714.0)]
        if len(valid_rr) == 0:
            return None

        mean_rr_s = np.mean(valid_rr) / 1000.0
        hr_bpm = 60.0 / mean_rr_s
        return float(np.round(hr_bpm, 1))

    def calculate_hrv(self, rr_intervals_ms: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        """
        Calculates standard HRV time-domain metrics:
        - SDNN (ms): Standard deviation of NN intervals
        - RMSSD (ms): Root mean square of successive differences
        - pNN50 (%): Percentage of successive NN intervals differing by > 50ms
        """
        if rr_intervals_ms is None or len(rr_intervals_ms) == 0:
            return {
                "sdnn_ms": None,
                "rmssd_ms": None,
                "pnn50_pct": None,
                "status": "Insufficient data (minimum 5 valid beats required)",
            }

        valid_rr = rr_intervals_ms[(rr_intervals_ms >= 272.0) & (rr_intervals_ms <= 1714.0)]
        if len(valid_rr) < 5:
            return {
                "sdnn_ms": None,
                "rmssd_ms": None,
                "pnn50_pct": None,
                "status": "Insufficient data (minimum 5 valid beats required)",
            }

        # SDNN
        sdnn = float(np.std(valid_rr, ddof=1))

        # RMSSD & pNN50 require successive differences
        diff_rr = np.diff(valid_rr)
        rmssd = float(np.sqrt(np.mean(diff_rr**2)))
        pnn50 = float((np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr)) * 100.0)

        return {
            "sdnn_ms": round(sdnn, 2),
            "rmssd_ms": round(rmssd, 2),
            "pnn50_pct": round(pnn50, 2),
            "status": "VALID",
        }

    def assess_signal_quality(
        self, raw_ecg: Optional[np.ndarray], leads_off: int = 0
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Evaluates ECG signal quality based on hardware lead-off, clipping, and SNR.
        """
        if leads_off == 1:
            return {
                "quality": "DISCONNECTED",
                "is_valid": False,
                "reason": "Hardware Lead-off detected (Electrode detached)",
                "snr_db": 0.0,
            }

        if raw_ecg is None:
            return {"quality": "INSUFFICIENT_DATA", "is_valid": False, "reason": "No data", "snr_db": 0.0}

        data = np.asarray(raw_ecg, dtype=np.float64)
        if len(data) < int(self.fs):
            return {"quality": "INSUFFICIENT_DATA", "is_valid": False, "reason": "Buffer too short", "snr_db": 0.0}

        # Check for ADC saturation / rail clipping (10-bit ADC: 0 or 1023)
        clipped_samples = np.sum((data <= 5) | (data >= 1018))
        clipping_ratio = clipped_samples / len(data)
        if clipping_ratio > 0.08:
            return {
                "quality": "POOR",
                "is_valid": False,
                "reason": f"Sensor clipping/saturation ({clipping_ratio*100:.1f}% samples clipped)",
                "snr_db": 0.0,
            }

        # Check for flatline / negligible amplitude
        if np.ptp(data) < 5.0 or np.std(data) < 1.0:
            return {
                "quality": "FLATLINE",
                "is_valid": False,
                "reason": "Flatline signal detected (negligible amplitude)",
                "snr_db": 0.0,
            }

        # Check baseline drift vs QRS power (SNR estimation)
        filtered = self.filter_signal(data)
        noise = data - filtered - np.mean(data)
        p_signal = np.var(filtered)
        p_noise = np.var(noise)

        if p_signal < 1e-4:
            snr_db = 0.0
        elif p_noise < 1e-6:
            snr_db = 30.0
        else:
            snr_db = float(10.0 * np.log10(p_signal / p_noise))

        if snr_db > 12.0:
            quality = "EXCELLENT"
            is_valid = True
        elif snr_db > 6.0:
            quality = "GOOD"
            is_valid = True
        else:
            quality = "POOR"
            is_valid = False

        return {
            "quality": quality,
            "is_valid": is_valid,
            "reason": f"Estimated SNR: {snr_db:.1f} dB",
            "snr_db": round(snr_db, 1),
        }

    def process(self, raw_ecg: np.ndarray, leads_off: int = 0) -> Dict[str, Any]:
        """Runs the complete ECG analysis pipeline on a window of samples."""
        sqi = self.assess_signal_quality(raw_ecg, leads_off=leads_off)
        if not sqi["is_valid"]:
            return {
                "filtered_ecg": raw_ecg,
                "r_peaks": np.array([], dtype=int),
                "rr_intervals_ms": np.array([], dtype=float),
                "heart_rate": None,
                "hrv": {"sdnn_ms": None, "rmssd_ms": None, "pnn50_pct": None, "status": sqi["reason"]},
                "signal_quality": sqi,
            }

        filtered = self.filter_signal(raw_ecg)
        r_peaks = self.detect_r_peaks(filtered)
        rr_ms = self.calculate_rr_intervals(r_peaks)
        hr = self.calculate_heart_rate(rr_ms)
        hrv = self.calculate_hrv(rr_ms)

        if len(r_peaks) < 2:
            hrv["status"] = "Missing pulses (fewer than 2 R-peaks detected)"

        return {
            "filtered_ecg": filtered,
            "r_peaks": r_peaks,
            "rr_intervals_ms": rr_ms,
            "heart_rate": hr,
            "hrv": hrv,
            "signal_quality": sqi,
        }
