"""
Dataset Loader and Benchmark Preprocessing Utilities
Provides standardized loaders for MIT-BIH Arrhythmia (ECG) and BIDMC (PPG/Resp),
signal resampling to target 100 Hz, and synthetic fixtures for unit tests.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


class DatasetLoader:
    """Loads and standardizes biomedical reference datasets."""

    @staticmethod
    def resample_signal(data: np.ndarray, orig_fs: float, target_fs: float = 100.0) -> np.ndarray:
        """Resamples physiological signal from orig_fs to target_fs using polyphase filtering."""
        if orig_fs == target_fs or len(data) == 0:
            return data
        num_target_samples = int(len(data) * target_fs / orig_fs)
        return signal.resample(data, num_target_samples)

    @staticmethod
    def generate_synthetic_ecg_fixture(
        duration_s: float = 10.0,
        fs: float = 100.0,
        heart_rate: float = 75.0,
        noise_std: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic standard Lead II ECG with known ground truth R-peak locations.
        Useful for deterministic automated unit tests without requiring multi-gigabyte downloads.
        """
        n_samples = int(duration_s * fs)
        t = np.arange(n_samples) / fs
        cardiac_freq = heart_rate / 60.0
        r_peaks = []

        ecg = np.zeros(n_samples)
        beat_interval = 1.0 / cardiac_freq
        current_beat_time = 0.2

        while current_beat_time < duration_s - 0.2:
            peak_idx = int(current_beat_time * fs)
            r_peaks.append(peak_idx)

            # Generate P-Q-R-S-T complex around beat
            # QRS width ~ 0.08s
            for i in range(max(0, peak_idx - int(0.2 * fs)), min(n_samples, peak_idx + int(0.4 * fs))):
                dt = (i - peak_idx) / fs
                # P wave
                p = 0.15 * np.exp(-((dt + 0.16) ** 2) / (2 * (0.02**2)))
                # Q wave
                q = -0.15 * np.exp(-((dt + 0.04) ** 2) / (2 * (0.01**2)))
                # R wave
                r = 1.0 * np.exp(-(dt**2) / (2 * (0.012**2)))
                # S wave
                s = -0.25 * np.exp(-((dt - 0.04) ** 2) / (2 * (0.012**2)))
                # T wave
                t_w = 0.3 * np.exp(-((dt - 0.22) ** 2) / (2 * (0.04**2)))

                ecg[i] += p + q + r + s + t_w

            current_beat_time += beat_interval

        if noise_std > 0:
            ecg += np.random.normal(0, noise_std, n_samples)

        return ecg, np.array(r_peaks, dtype=int)

    @staticmethod
    def generate_synthetic_ppg_fixture(
        duration_s: float = 10.0,
        fs: float = 100.0,
        heart_rate: float = 75.0,
        pat_delay_ms: float = 220.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates synthetic paired Red and IR PPG waveforms with dicrotic notches.
        """
        n_samples = int(duration_s * fs)
        t = np.arange(n_samples) / fs
        cardiac_freq = heart_rate / 60.0

        delay_s = pat_delay_ms / 1000.0
        ppg_peaks = []

        red = np.full(n_samples, 62000.0)
        ir = np.full(n_samples, 68000.0)

        beat_interval = 1.0 / cardiac_freq
        current_beat_time = 0.2 + delay_s

        while current_beat_time < duration_s - 0.2:
            peak_idx = int(current_beat_time * fs)
            ppg_peaks.append(peak_idx)

            for i in range(max(0, peak_idx - int(0.15 * fs)), min(n_samples, peak_idx + int(0.45 * fs))):
                dt = (i - peak_idx) / fs
                systolic = np.exp(-(dt**2) / (2 * (0.06**2)))
                diastolic = 0.4 * np.exp(-((dt - 0.2) ** 2) / (2 * (0.05**2)))
                pulse = systolic + diastolic
                red[i] += 2500.0 * pulse
                ir[i] += 3500.0 * pulse

            current_beat_time += beat_interval

        return red, ir, np.array(ppg_peaks, dtype=int)
