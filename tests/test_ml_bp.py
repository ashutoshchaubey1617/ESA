"""
Unit tests for Machine Learning Blood Pressure Estimation Engine
"""

import numpy as np
import pytest
from ml.bp_model import BloodPressureFeatureExtractor, BloodPressureModel


def test_bp_feature_extraction():
    extractor = BloodPressureFeatureExtractor(sampling_rate=100.0)

    # Paired R-peaks and PPG peaks with 200ms PAT
    r_peaks = np.array([20, 100, 180, 260, 340, 420])
    ppg_peaks = np.array([40, 120, 200, 280, 360, 440])

    ecg = np.random.randn(500)
    ppg = np.random.randn(500)
    raw_ir = np.full(500, 68000.0)

    features = extractor.extract_features(r_peaks, ecg, ppg_peaks, ppg, raw_ir)
    assert features is not None
    assert "pat_ms" in features
    assert "hr_bpm" in features
    assert abs(features["pat_ms"] - 200.0) < 1.0


def test_bp_model_uncalibrated_gating():
    model = BloodPressureModel(model_file=None)
    assert model.is_trained is False

    dummy_features = {
        "pat_ms": 220.0,
        "hr_bpm": 72.0,
        "rise_time_ms": 110.0,
        "pulse_width_ms": 210.0,
        "perfusion_index": 1.5,
    }
    pred = model.predict(dummy_features)
    assert pred["status"] == "UNCALIBRATED"
    assert pred["sbp_mmHg"] is None
    assert "disabled" in pred["disclaimer"].lower()


def test_bp_model_training_and_evaluation():
    model = BloodPressureModel(model_file=None)

    # Generate synthetic training set: 100 physiological observations
    np.random.seed(42)
    n_samples = 100
    pat = np.random.uniform(150, 300, n_samples)
    hr = np.random.uniform(60, 100, n_samples)
    rise = np.random.uniform(80, 150, n_samples)
    width = np.random.uniform(180, 250, n_samples)
    pi = np.random.uniform(1.0, 3.0, n_samples)

    X = np.column_stack([pat, hr, rise, width, pi])
    # Physiological inverse relationship between PAT and BP (Moens-Korteweg equation)
    y_sbp = 180.0 - 0.25 * pat + 0.2 * hr + np.random.normal(0, 3, n_samples)
    y_dbp = 110.0 - 0.15 * pat + 0.1 * hr + np.random.normal(0, 2, n_samples)

    metrics = model.train(X, y_sbp, y_dbp, test_size=0.2)

    assert model.is_trained is True
    assert "sbp" in metrics
    assert "dbp" in metrics
    assert metrics["sbp"]["mae_mmHg"] < 10.0
    assert metrics["dbp"]["mae_mmHg"] < 8.0

    # Predict test
    pred = model.predict(
        {
            "pat_ms": 200.0,
            "hr_bpm": 75.0,
            "rise_time_ms": 110.0,
            "pulse_width_ms": 210.0,
            "perfusion_index": 1.5,
        }
    )
    assert pred["status"] == "ESTIMATED_RESEARCH_ONLY"
    assert pred["sbp_mmHg"] is not None
    assert pred["dbp_mmHg"] is not None
