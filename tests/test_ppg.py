"""
Unit tests for MAX30102 PPG & SpO2 Processing Module
"""

import numpy as np
import pytest
from biomedical.ppg import PPGProcessor
from datasets.data_loader import DatasetLoader


@pytest.fixture
def ppg_proc():
    return PPGProcessor(sampling_rate=100.0)


def test_finger_contact_detection(ppg_proc):
    # Case 1: Low values (no finger)
    low_ir = np.full(100, 5000.0)
    low_red = np.full(100, 4000.0)
    res_no_contact = ppg_proc.check_finger_contact(low_red, low_ir)
    assert res_no_contact["contact"] is False

    # Case 2: Good optical return
    good_ir = np.full(100, 68000.0)
    good_red = np.full(100, 62000.0)
    res_contact = ppg_proc.check_finger_contact(good_red, good_ir)
    assert res_contact["contact"] is True


def test_ppg_processing_and_spo2(ppg_proc):
    red, ir, ground_truth_peaks = DatasetLoader.generate_synthetic_ppg_fixture(
        duration_s=8.0, fs=100.0, heart_rate=75.0, pat_delay_ms=220.0
    )

    result = ppg_proc.process(red, ir)

    assert result["contact_status"]["contact"] is True
    assert len(result["pulse_peaks"]) >= 7

    # SpO2 should be in clinical range [90%, 100%]
    assert result["spo2"] is not None
    assert 90.0 <= result["spo2"] <= 100.0

    # Heart rate should be ~75 BPM
    assert result["heart_rate"] is not None
    assert abs(result["heart_rate"] - 75.0) <= 3.0


def test_ppg_saturation(ppg_proc):
    sat_red = np.full(200, 260000.0)
    sat_ir = np.full(200, 260000.0)
    contact = ppg_proc.check_finger_contact(sat_red, sat_ir)
    assert contact["contact"] is False
    assert "saturated" in contact["reason"].lower()

    res = ppg_proc.process(sat_red, sat_ir)
    assert res["heart_rate"] is None
    assert res["spo2"] is None


def test_ppg_flatline(ppg_proc):
    flat_red = np.full(500, 60000.0)
    flat_ir = np.full(500, 60000.0)
    # Detect pulse peaks on flatline returns empty
    filtered_ir = ppg_proc.filter_ppg(flat_ir)
    peaks = ppg_proc.detect_pulse_peaks(filtered_ir)
    assert len(peaks) == 0
    # SpO2 on flatline returns None
    filtered_red = ppg_proc.filter_ppg(flat_red)
    spo2 = ppg_proc.estimate_spo2(flat_red, flat_ir, filtered_red, filtered_ir)
    assert spo2 is None


def test_ppg_missing_pulses_and_gating(ppg_proc):
    # Only 1 peak
    single_peak = np.array([40])
    assert ppg_proc.calculate_heart_rate(single_peak) is None

    # None peak
    assert ppg_proc.calculate_heart_rate(None) is None

    # Bradycardia / Tachycardia gating
    # 25 BPM: peak every 240 samples at 100Hz = 2400ms (> 1714ms)
    brady_peaks = np.array([0, 240, 480])
    assert ppg_proc.calculate_heart_rate(brady_peaks) is None

    # 250 BPM: peak every 24 samples at 100Hz = 240ms (< 272ms)
    tachy_peaks = np.array([0, 24, 48, 72])
    assert ppg_proc.calculate_heart_rate(tachy_peaks) is None

