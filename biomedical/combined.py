"""
Multimodal Biomedical Analysis Module
Implements ECG-versus-PPG comparison, Pulse Arrival Time (PAT) computation,
and Respiratory Rate (RR) estimation from cardiac signals.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import signal


class CombinedBiomedicalAnalyzer:
    """Combines synchronized ECG and PPG signals for advanced physiological metrics."""

    def __init__(self, sampling_rate: float = 100.0):
        self.fs = float(sampling_rate)

    def compare_heart_rates(
        self, ecg_hr: Optional[float], ppg_hr: Optional[float]
    ) -> Dict[str, Any]:
        """
        Compares ECG-derived and PPG-derived heart rates for concordance.
        Discrepancies > 10 BPM indicate motion artifacts, arrhythmia, or poor sensor contact.
        """
        if ecg_hr is None or ppg_hr is None or np.isnan(ecg_hr) or np.isnan(ppg_hr):
            return {
                "difference_bpm": None,
                "is_concordant": False,
                "status": "Incomplete data (one or both HR unavailable)",
            }

        diff = abs(ecg_hr - ppg_hr)
        is_concordant = diff <= 8.0

        status = "EXCELLENT" if diff <= 3.0 else ("ACCEPTABLE" if diff <= 8.0 else "DISCREPANCY_DETECTED")

        return {
            "difference_bpm": round(diff, 1),
            "is_concordant": is_concordant,
            "status": status,
        }

    def compute_pulse_arrival_time(
        self,
        r_peaks: Optional[np.ndarray],
        ppg_peaks: Optional[np.ndarray],
        min_pat_ms: float = 120.0,
        max_pat_ms: float = 450.0,
    ) -> Dict[str, Any]:
        """
        Estimates Pulse Arrival Time (PAT) in milliseconds.

        SCIENTIFIC NOTE & DISTINCTION:
        - Pulse Arrival Time (PAT) is the interval between electrical ventricular depolarization
          (ECG R-peak) and the arrival of the mechanical pressure pulse at the peripheral site (PPG peak).
        - PAT = PEP + PTT, where:
            * PEP (Pre-Ejection Period): Electromechanical delay and isovolumetric contraction time.
            * PTT (Pulse Transit Time): Actual arterial travel time through vascular tree.
        Because surface sensors without phonocardiography or impedance cardiography cannot directly
        measure the aortic valve opening, this method measures PAT.
        """
        if r_peaks is None or ppg_peaks is None or len(r_peaks) < 2 or len(ppg_peaks) < 2:
            return {
                "pat_median_ms": None,
                "pat_std_ms": None,
                "valid_pairs_count": 0,
                "status": "Insufficient peaks for PAT calculation",
            }

        pat_values_ms = []

        # For each R-peak, locate the immediately following PPG peak
        for r_idx in r_peaks:
            # Subsequent PPG peaks
            subsequent_ppg = ppg_peaks[ppg_peaks > r_idx]
            if len(subsequent_ppg) == 0:
                continue

            nearest_ppg = subsequent_ppg[0]
            delay_samples = nearest_ppg - r_idx
            delay_ms = (delay_samples / self.fs) * 1000.0

            # Plausible physiological window for finger/wrist PAT: 120ms to 450ms
            if min_pat_ms <= delay_ms <= max_pat_ms:
                pat_values_ms.append(delay_ms)

        if len(pat_values_ms) < 2:
            return {
                "pat_median_ms": None,
                "pat_std_ms": None,
                "valid_pairs_count": len(pat_values_ms),
                "status": "Poor peak synchronization or missing pulses",
            }

        median_pat = float(np.median(pat_values_ms))
        std_pat = float(np.std(pat_values_ms))

        return {
            "pat_median_ms": round(median_pat, 1),
            "pat_std_ms": round(std_pat, 1),
            "valid_pairs_count": len(pat_values_ms),
            "status": "VALID",
        }

    def estimate_respiratory_rate(
        self,
        filtered_ecg: Optional[np.ndarray],
        r_peaks: Optional[np.ndarray],
        filtered_ppg: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Estimates Respiratory Rate (Breaths per Minute) from:
        1. ECG-Derived Respiration (EDR) via R-peak amplitude modulation
        2. PPG Respiratory-Induced Amplitude Modulation (RIAM) fallback
        Requires at least 12 seconds of valid data to resolve 0.1 - 0.5 Hz (6 - 30 BrPM).
        """
        min_samples = int(self.fs * 12.0)
        if (
            filtered_ecg is None
            or r_peaks is None
            or len(filtered_ecg) < min_samples
            or len(r_peaks) < 8
        ):
            return {
                "respiratory_rate_bpm": None,
                "method": "N/A",
                "status": "Insufficient window duration (minimum 12s required for respiration)",
            }

        # Method 1: EDR via R-peak amplitude modulation
        try:
            r_amplitudes = filtered_ecg[r_peaks]
            # Verify amplitude modulation is real, not flatline
            if np.ptp(r_amplitudes) >= 1e-4 and np.std(r_amplitudes) >= 1e-4:
                r_times = r_peaks / self.fs

                # Interpolate to uniform 4 Hz grid
                target_fs = 4.0
                t_uniform = np.arange(r_times[0], r_times[-1], 1.0 / target_fs)
                if len(t_uniform) >= 16:
                    edr_signal = np.interp(t_uniform, r_times, r_amplitudes)
                    edr_signal = edr_signal - np.mean(edr_signal)

                    # Bandpass filter for respiration: 0.1 Hz to 0.5 Hz (6 to 30 breaths per min)
                    nyq = 0.5 * target_fs
                    b, a = signal.butter(2, [0.1 / nyq, 0.5 / nyq], btype="bandpass")
                    resp_filtered = signal.filtfilt(b, a, edr_signal)

                    if np.std(resp_filtered) >= 1e-4:
                        # Compute power spectral density using Welch
                        freqs, psd = signal.welch(resp_filtered, fs=target_fs, nperseg=min(len(resp_filtered), 64))
                        valid_band = (freqs >= 0.1) & (freqs <= 0.5)

                        if np.any(valid_band) and np.max(psd[valid_band]) >= 1e-6:
                            peak_freq = freqs[valid_band][np.argmax(psd[valid_band])]
                            resp_bpm = float(peak_freq * 60.0)

                            if 6.0 <= resp_bpm <= 35.0:
                                return {
                                    "respiratory_rate_bpm": round(resp_bpm, 1),
                                    "method": "ECG-Derived Respiration (EDR)",
                                    "status": "VALID",
                                }
        except Exception:
            pass

        # Method 2: Fallback to PPG RIAM if PPG waveform is available
        if filtered_ppg is not None and len(filtered_ppg) >= min_samples:
            try:
                ppg_data = np.asarray(filtered_ppg, dtype=np.float64)
                if np.std(ppg_data) >= 1.0:
                    # Resample PPG to 4 Hz
                    target_fs = 4.0
                    num_target = int(len(ppg_data) * target_fs / self.fs)
                    ppg_downsampled = signal.resample(ppg_data, num_target)
                    ppg_downsampled = ppg_downsampled - np.mean(ppg_downsampled)

                    nyq = 0.5 * target_fs
                    b, a = signal.butter(2, [0.1 / nyq, 0.5 / nyq], btype="bandpass")
                    resp_ppg = signal.filtfilt(b, a, ppg_downsampled)

                    if np.std(resp_ppg) >= 1e-4:
                        freqs, psd = signal.welch(resp_ppg, fs=target_fs, nperseg=min(len(resp_ppg), 64))
                        valid_band = (freqs >= 0.1) & (freqs <= 0.5)
                        if np.any(valid_band) and np.max(psd[valid_band]) >= 1e-6:
                            peak_freq = freqs[valid_band][np.argmax(psd[valid_band])]
                            resp_bpm = float(peak_freq * 60.0)
                            if 6.0 <= resp_bpm <= 35.0:
                                return {
                                    "respiratory_rate_bpm": round(resp_bpm, 1),
                                    "method": "PPG-Derived Respiration (RIAM)",
                                    "status": "VALID",
                                }
            except Exception:
                pass

        return {
            "respiratory_rate_bpm": None,
            "method": "EDR",
            "status": "Could not extract clear respiratory rhythm",
        }
