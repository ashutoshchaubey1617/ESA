"""
Unit tests for ECG Signal Processing Module
"""

import numpy as np
import pytest
from biomedical.ecg import ECGProcessor
from datasets.data_loader import DatasetLoader


@pytest.fixture
def ecg_proc():
    return ECGProcessor(sampling_rate=100.0)


def test_ecg_filtering(ecg_proc):
    data, _ = DatasetLoader.generate_synthetic_ecg_fixture(duration_s=5.0, fs=100.0, heart_rate=75.0)
    filtered = ecg_proc.filter_signal(data)
    assert len(filtered) == len(data)
    assert not np.isnan(filtered).any()
    assert abs(np.mean(filtered)) < 0.1  # DC baseline removed


def test_r_peak_detection(ecg_proc):
    data, ground_truth_peaks = DatasetLoader.generate_synthetic_ecg_fixture(
        duration_s=8.0, fs=100.0, heart_rate=72.0, noise_std=0.01
    )
    filtered = ecg_proc.filter_signal(data)
    detected_peaks = ecg_proc.detect_r_peaks(filtered)

    assert len(detected_peaks) >= 8
    # Check that detected peaks align closely with ground truth peaks (within 2 samples = 20ms)
    matches = 0
    for gt in ground_truth_peaks:
        if np.any(np.abs(detected_peaks - gt) <= 2):
            matches += 1
    sensitivity = matches / len(ground_truth_peaks)
    assert sensitivity >= 0.85


def test_heart_rate_and_hrv(ecg_proc):
    # 75 BPM -> beat interval = 60/75 = 0.8s = 800ms
    r_peaks = np.array([20, 100, 180, 260, 340, 420, 500, 580, 660])  # 80 sample intervals = 800ms at 100Hz
    rr_ms = ecg_proc.calculate_rr_intervals(r_peaks)

    assert len(rr_ms) == len(r_peaks) - 1
    assert np.allclose(rr_ms, 800.0, atol=1.0)

    hr = ecg_proc.calculate_heart_rate(rr_ms)
    assert hr is not None
    assert abs(hr - 75.0) < 1.0

    hrv = ecg_proc.calculate_hrv(rr_ms)
    assert hrv["status"] == "VALID"
    assert hrv["rmssd_ms"] is not None
    assert hrv["sdnn_ms"] is not None


def test_signal_quality_lead_off(ecg_proc):
    raw = np.full(200, 1023)
    sqi_lead_off = ecg_proc.assess_signal_quality(raw, leads_off=1)
    assert sqi_lead_off["quality"] == "DISCONNECTED"
    assert sqi_lead_off["is_valid"] is False

    sqi_clipped = ecg_proc.assess_signal_quality(raw, leads_off=0)
    assert sqi_clipped["quality"] == "POOR"
    assert "clipping" in sqi_clipped["reason"].lower()


def test_ecg_flatline(ecg_proc):
    flatline = np.full(500, 512.0)
    sqi = ecg_proc.assess_signal_quality(flatline)
    assert sqi["quality"] == "FLATLINE"
    assert sqi["is_valid"] is False

    res = ecg_proc.process(flatline)
    assert len(res["r_peaks"]) == 0
    assert res["heart_rate"] is None
    assert "flatline" in res["hrv"]["status"].lower()


def test_ecg_missing_pulses(ecg_proc):
    # Only 1 peak
    r_peaks_single = np.array([50])
    rr = ecg_proc.calculate_rr_intervals(r_peaks_single)
    assert len(rr) == 0
    assert ecg_proc.calculate_heart_rate(rr) is None
    hrv = ecg_proc.calculate_hrv(rr)
    assert hrv["rmssd_ms"] is None


def test_ecg_physiological_gating(ecg_proc):
    # Extreme bradycardia: 25 BPM -> interval = 2400ms (> 1714ms limit)
    rr_brady = np.array([2400.0, 2400.0, 2400.0])
    assert ecg_proc.calculate_heart_rate(rr_brady) is None

    # Extreme tachycardia: 250 BPM -> interval = 240ms (< 272ms limit)
    rr_tachy = np.array([240.0, 240.0, 240.0])
    assert ecg_proc.calculate_heart_rate(rr_tachy) is None

    # Plausible rate: 60 BPM -> 1000ms
    rr_normal = np.array([1000.0, 1000.0, 1000.0])
    assert ecg_proc.calculate_heart_rate(rr_normal) == 60.0


def test_ecg_notch_filter():
    # 250 Hz sampling rate: Nyquist = 125 Hz, notch at 50 Hz
    proc250 = ECGProcessor(sampling_rate=250.0)
    t = np.arange(500) / 250.0
    # 50 Hz powerline interference + 1.2 Hz ECG-like wave
    noise_50hz = 200.0 * np.sin(2 * np.pi * 50.0 * t)
    carrier = 300.0 * np.sin(2 * np.pi * 1.2 * t)
    raw = carrier + noise_50hz
    filtered = proc250.filter_signal(raw, notch_freq=50.0)
    # 50 Hz interference must be attenuated by at least 15 dB
    fft_raw = np.abs(np.fft.rfft(raw))
    fft_filtered = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(raw), 1.0 / 250.0)
    idx_50 = np.argmin(np.abs(freqs - 50.0))
    attenuation_ratio = fft_filtered[idx_50] / (fft_raw[idx_50] + 1e-6)
    assert attenuation_ratio < 0.1

