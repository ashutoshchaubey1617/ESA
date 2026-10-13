"""
Unit tests for Combined Biomedical Analyzer (PAT, Concordance, Respiration)
"""

import numpy as np
import pytest
from biomedical.combined import CombinedBiomedicalAnalyzer
from datasets.data_loader import DatasetLoader


@pytest.fixture
def combined():
    return CombinedBiomedicalAnalyzer(sampling_rate=100.0)


def test_heart_rate_comparison(combined):
    comp_good = combined.compare_heart_rates(72.0, 73.5)
    assert comp_good["is_concordant"] is True
    assert comp_good["difference_bpm"] == 1.5

    comp_divergent = combined.compare_heart_rates(72.0, 92.0)
    assert comp_divergent["is_concordant"] is False
    assert comp_divergent["difference_bpm"] == 20.0


def test_pulse_arrival_time(combined):
    # ECG peaks at 0, 80, 160, 240... (every 800ms)
    # PPG peaks delayed by 22 samples = 220ms -> 22, 102, 182, 262...
    r_peaks = np.array([0, 80, 160, 240, 320, 400])
    ppg_peaks = np.array([22, 102, 182, 262, 342, 422])

    pat_res = combined.compute_pulse_arrival_time(r_peaks, ppg_peaks)
    assert pat_res["status"] == "VALID"
    assert pat_res["pat_median_ms"] == 220.0
    assert pat_res["valid_pairs_count"] == 6


def test_respiration_rate_insufficient_window(combined):
    short_ecg = np.zeros(200)
    short_r = np.array([20, 100])
    short_ppg = np.zeros(200)

    res = combined.estimate_respiratory_rate(short_ecg, short_r, short_ppg)
    assert res["respiratory_rate_bpm"] is None
    assert "insufficient" in res["status"].lower()


def test_heart_rate_comparison_edge_cases(combined):
    res_none = combined.compare_heart_rates(None, 75.0)
    assert res_none["is_concordant"] is False
    assert res_none["difference_bpm"] is None

    res_nan = combined.compare_heart_rates(75.0, float("nan"))
    assert res_nan["is_concordant"] is False
    assert res_nan["difference_bpm"] is None


def test_pulse_arrival_time_edge_cases(combined):
    # None inputs
    assert combined.compute_pulse_arrival_time(None, np.array([20, 100]))["status"] == "Insufficient peaks for PAT calculation"

    # Fewer than 2 peaks
    assert combined.compute_pulse_arrival_time(np.array([20]), np.array([50, 120]))["status"] == "Insufficient peaks for PAT calculation"

    # Out of physiological bounds (< 120ms or > 450ms)
    # Delay = 10ms (1 sample)
    r_peaks = np.array([100, 200, 300])
    ppg_too_early = np.array([101, 201, 301])  # 10ms delay
    res_early = combined.compute_pulse_arrival_time(r_peaks, ppg_too_early)
    assert res_early["pat_median_ms"] is None
    assert "missing pulses" in res_early["status"].lower() or "poor peak" in res_early["status"].lower()


def test_respiration_rate_flatline(combined):
    # 15s flatline ECG
    flat_ecg = np.full(1500, 512.0)
    r_peaks = np.arange(50, 1400, 80)
    res = combined.estimate_respiratory_rate(flat_ecg, r_peaks)
    assert res["respiratory_rate_bpm"] is None


def test_respiration_rate_modulated(combined):
    # 16 seconds at 100Hz = 1600 samples
    # 75 BPM -> beat every 80 samples (0.8s) -> 20 beats
    # Respiratory rate = 15 BrPM = 0.25 Hz (period = 4.0s)
    fs = 100.0
    duration_s = 16.0
    n = int(duration_s * fs)
    ecg = np.zeros(n)
    r_peaks = []
    for beat_time in np.arange(0.5, duration_s - 0.5, 0.8):
        idx = int(beat_time * fs)
        r_peaks.append(idx)
        # Respiration amplitude modulation (0.25 Hz = 15 BrPM)
        amp = 1.0 + 0.3 * np.sin(2 * np.pi * 0.25 * beat_time)
        ecg[idx] = amp

    res = combined.estimate_respiratory_rate(ecg, np.array(r_peaks), None)
    assert res["status"] == "VALID"
    assert res["respiratory_rate_bpm"] is not None
    # 0.25 Hz * 60 = 15 BrPM (allow resolution window tolerance)
    assert abs(res["respiratory_rate_bpm"] - 15.0) <= 2.5

