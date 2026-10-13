"""
Verification script for Biometric Enrollment and Blood Pressure ML Calibration.
Validates:
1. datasets/face_features/feature.npz integrity
2. SQLite health_monitor.db participant records for Ashutosh and Dhrunil
3. FaceService live recognition accuracy and out-of-the-box enrollment
4. BloodPressureModel loading from models/rf_bp_model.joblib, AAMI compliance, and physiological predictions
"""

import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np

from database.db import DatabaseManager
from face_pipeline.face_service import FaceService
from face_recognition.arcface.utils import read_features
from ml.bp_model import BloodPressureModel


def test_feature_npz():
    print("--- 1. Validating datasets/face_features/feature.npz ---")
    feat_file = "datasets/face_features/feature.npz"
    assert os.path.exists(feat_file), f"Missing {feat_file}"

    names, embs = read_features("datasets/face_features/feature")
    print(f"Total entries in feature.npz: {len(names)}")
    assert len(names) >= 21, f"Expected >= 21 entries, found {len(names)}"
    assert len(embs) == len(names), "Names and embeddings count mismatch"

    embs_arr = np.array(embs)
    assert embs_arr.shape[1] == 512, f"Expected 512-dim embeddings, got {embs_arr.shape[1]}"

    # Verify unit norm of all embeddings
    norms = np.linalg.norm(embs_arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not properly L2-normalized"

    ash_count = sum(1 for n in names if n == "Ashutosh")
    dhr_count = sum(1 for n in names if n == "Dhrunil")
    print(f"Ashutosh face count: {ash_count} (Expected: 10)")
    print(f"Dhrunil face count: {dhr_count} (Expected: 7)")
    assert ash_count == 10, f"Expected 10 Ashutosh faces, got {ash_count}"
    assert dhr_count == 7, f"Expected 7 Dhrunil faces, got {dhr_count}"
    print("[PASS] feature.npz integrity validated.\n")


def test_sqlite_participants():
    print("--- 2. Validating SQLite health_monitor.db ---")
    db = DatabaseManager("health_monitor.db")
    participants = db.get_all_participants()
    p_dict = {p["name"]: p for p in participants}

    print(f"Enrolled participants in database: {list(p_dict.keys())}")
    assert "Ashutosh" in p_dict, "Ashutosh not found in health_monitor.db participants"
    assert "Dhrunil" in p_dict, "Dhrunil not found in health_monitor.db participants"

    for name in ["Ashutosh", "Dhrunil"]:
        p = p_dict[name]
        emb = p.get("embedding")
        assert emb is not None, f"Missing embedding vector for {name}"
        assert emb.shape == (512,), f"Expected shape (512,), got {emb.shape} for {name}"
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-3, f"Averaged embedding for {name} is not unit normalized: {norm}"
        print(f"  {name}: ID={p['participant_id']}, EnrolledAt={p['enrolled_at']}, Norm={norm:.4f}")
        print(f"  Notes: {p['notes']}")

    print("[PASS] SQLite database participants validated.\n")


def test_face_service_recognition():
    print("--- 3. Validating FaceService Out-of-the-Box Recognition ---")
    db = DatabaseManager("health_monitor.db")
    face_srv = FaceService(db)

    assert face_srv.is_ready is True, f"FaceService not ready: {face_srv.status_message}"
    assert "Ashutosh" in face_srv.known_names, "Ashutosh missing from FaceService cache"
    assert "Dhrunil" in face_srv.known_names, "Dhrunil missing from FaceService cache"
    print(f"FaceService cache: {face_srv.known_names}")
    print(f"Known embeddings tensor shape: {face_srv.known_embeddings.shape}")

    # Test detection and identification on multiple raw images
    test_cases = [
        ("Ashutosh", "datasets/data/ashutosh/0.jpg", 0.85),
        ("Ashutosh", "datasets/data/ashutosh/3.jpg", 0.85),
        ("Dhrunil", "datasets/data/dhrunil/0.jpg", 0.80),
        ("Dhrunil", "datasets/data/dhrunil/4.jpg", 0.80),
    ]

    for expected_name, img_path, min_sim in test_cases:
        img = cv2.imread(img_path)
        assert img is not None, f"Could not load {img_path}"
        res = face_srv.process_frame(img)
        detections = res.get("detections", [])
        assert len(detections) > 0, f"No face detected in {img_path}"

        det = detections[0]
        print(f"Image: {img_path}")
        print(f"  Identified: {det['name']} (Sim: {det['similarity']}, ID: {det['participant_id']})")
        assert det["name"] == expected_name, f"Expected {expected_name}, got {det['name']}"
        assert det["is_known"] is True, f"Expected is_known=True for {expected_name}"
        assert det["similarity"] >= min_sim, f"Similarity {det['similarity']} < {min_sim}"

    print("[PASS] FaceService recognition validated.\n")


def test_bp_model_loading_and_prediction():
    print("--- 4. Validating BloodPressureModel and Calibration Weights ---")
    model_file = "models/rf_bp_model.joblib"
    assert os.path.exists(model_file), f"Model weights missing at {model_file}"

    # Default constructor loads models/rf_bp_model.joblib
    bp = BloodPressureModel()
    assert bp.is_trained is True, "BloodPressureModel failed to load trained weights"
    assert bp.sbp_model is not None, "SBP model is None"
    assert bp.dbp_model is not None, "DBP model is None"

    metrics = bp.evaluation_metrics
    print("Model Evaluation Metrics:")
    print(f"  SBP: MAE={metrics['sbp']['mae_mmHg']} mmHg, RMSE={metrics['sbp']['rmse_mmHg']} mmHg, "
          f"MeanErr={metrics['sbp']['mean_error']} mmHg, StdErr={metrics['sbp']['std_error']} mmHg, "
          f"AAMI={metrics['sbp']['aami_compliant']}")
    print(f"  DBP: MAE={metrics['dbp']['mae_mmHg']} mmHg, RMSE={metrics['dbp']['rmse_mmHg']} mmHg, "
          f"MeanErr={metrics['dbp']['mean_error']} mmHg, StdErr={metrics['dbp']['std_error']} mmHg, "
          f"AAMI={metrics['dbp']['aami_compliant']}")

    assert metrics["sbp"]["aami_compliant"] is True, "SBP model does not meet AAMI SP10 criteria"
    assert metrics["dbp"]["aami_compliant"] is True, "DBP model does not meet AAMI SP10 criteria"

    # Physiological verification across hemodynamic states
    states = [
        (
            "Normotensive Baseline",
            {"pat_ms": 240.0, "hr_bpm": 72.0, "rise_time_ms": 110.0, "pulse_width_ms": 210.0, "perfusion_index": 2.0},
            (112.0, 124.0),
            (72.0, 80.0),
        ),
        (
            "High Arterial Stiffness / Sympathetic (Low PAT, High HR)",
            {"pat_ms": 170.0, "hr_bpm": 95.0, "rise_time_ms": 85.0, "pulse_width_ms": 240.0, "perfusion_index": 0.9},
            (135.0, 165.0),
            (85.0, 105.0),
        ),
        (
            "Vasodilation / High Elasticity (High PAT, Low HR)",
            {"pat_ms": 320.0, "hr_bpm": 58.0, "rise_time_ms": 130.0, "pulse_width_ms": 190.0, "perfusion_index": 3.2},
            (88.0, 110.0),
            (54.0, 72.0),
        ),
    ]

    for state_name, features, sbp_bounds, dbp_bounds in states:
        pred = bp.predict(features)
        sbp = pred["sbp_mmHg"]
        dbp = pred["dbp_mmHg"]
        status = pred["status"]
        print(f"State: {state_name}")
        print(f"  Prediction: SBP={sbp} mmHg (Expected {sbp_bounds}), DBP={dbp} mmHg (Expected {dbp_bounds}), Status={status}")
        assert status == "ESTIMATED_RESEARCH_ONLY", f"Unexpected status: {status}"
        assert sbp_bounds[0] <= sbp <= sbp_bounds[1], f"SBP {sbp} out of bounds {sbp_bounds}"
        assert dbp_bounds[0] <= dbp <= dbp_bounds[1], f"DBP {dbp} out of bounds {dbp_bounds}"
        assert sbp > dbp + 20.0, f"Physiologically invalid pulse pressure: SBP={sbp}, DBP={dbp}"

    print("[PASS] BloodPressureModel calibration and physiological predictions validated.\n")


def main():
    print("=================================================================")
    print("BIOMETRIC TRAINING & ENROLLMENT SYSTEM VERIFICATION SUITE")
    print("=================================================================\n")
    test_feature_npz()
    test_sqlite_participants()
    test_face_service_recognition()
    test_bp_model_loading_and_prediction()
    print("=================================================================")
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
