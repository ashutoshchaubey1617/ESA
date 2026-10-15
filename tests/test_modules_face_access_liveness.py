"""
Verification tests for modules/face_access.py and modules/liveness.py.
Validates:
1. Face detection & landmark extraction (SCRFD)
2. Enrollment dataset for Ashutosh (10 samples) and Dhrunil (7 samples)
3. Cosine similarity thresholding via ArcFace compare_encodings
4. Anti-spoofing liveness dynamics (live vs static photo attack vs extreme angle)
5. Unknown participant classification without false restrictions
6. vectornguyen76 module pipeline conformity (norm_crop 112x112, L2 normalization)
"""

import os
import tempfile
import cv2
import numpy as np
import pytest
import torch

from database.db import DatabaseManager
from modules.liveness import (
    LivenessDetector,
    check_geometric_proportions,
    estimate_head_pose,
    compute_liveness,
)
from modules.face_access import FaceAccessController
from face_recognition.arcface.utils import compare_encodings, read_features


@pytest.fixture
def temp_db():
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    db = DatabaseManager(temp_file.name)
    db.seed_default_participants()
    yield db
    try:
        os.remove(temp_file.name)
    except Exception:
        pass


def test_enrollment_dataset_integrity():
    """Verify enrolled dataset for Ashutosh and Dhrunil in feature.npz and SQLite."""
    feat_path = "datasets/face_features/feature"
    names, embs = read_features(feat_path)
    assert len(names) >= 21
    assert len(embs) == len(names)

    ash_count = sum(1 for n in names if str(n).strip().lower() == "ashutosh")
    dhr_count = sum(1 for n in names if str(n).strip().lower() == "dhrunil")

    assert ash_count == 10, f"Expected 10 Ashutosh samples, got {ash_count}"
    assert dhr_count == 7, f"Expected 7 Dhrunil samples, got {dhr_count}"

    # Verify L2 normalization of stored embeddings
    for emb in embs:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-3, f"Embedding norm {norm} is not ~1.0"


def test_liveness_geometric_proportions():
    """Verify facial geometry plausibility check according to biological ratios."""
    # Plausible frontal face
    valid_landmarks = np.array(
        [[100, 100], [200, 100], [150, 150], [115, 200], [185, 200]], dtype=np.float32
    )
    res = check_geometric_proportions(valid_landmarks)
    assert res["is_plausible"] is True
    assert 0.10 <= res["eye_mouth_ratio"] <= 4.2
    assert 0.15 <= res["mouth_eye_ratio"] <= 2.8

    # Warped geometry: mouth placed above the eyes (nose not between eyes and mouth)
    warped_landmarks = np.array(
        [[100, 100], [200, 100], [150, 150], [115, 50], [185, 50]], dtype=np.float32
    )
    res_warped = check_geometric_proportions(warped_landmarks)
    assert res_warped["is_plausible"] is False


def test_liveness_head_pose_estimation():
    """Verify 3D head pose estimation and extreme angle detection."""
    # Frontal
    frontal = np.array(
        [[100, 100], [200, 100], [150, 150], [115, 200], [185, 200]], dtype=np.float32
    )
    pose = estimate_head_pose(frontal)
    assert pose["is_extreme"] is False
    assert abs(pose["yaw"]) < 10.0
    assert abs(pose["roll"]) < 5.0

    # High yaw (> 32 deg)
    turned = np.array(
        [[100, 100], [200, 100], [195, 150], [120, 200], [190, 200]], dtype=np.float32
    )
    pose_turned = estimate_head_pose(turned)
    assert pose_turned["is_extreme"] is True
    assert pose_turned["yaw"] > 32.0


def test_liveness_micro_motion_anti_spoofing():
    """Verify detection of static photo attacks vs dynamic live subjects."""
    detector = LivenessDetector()
    base_lm = np.array(
        [[100, 100], [200, 100], [150, 150], [115, 200], [185, 200]], dtype=np.float32
    )

    # 1. Static photo attack: 15 identical frames with 0 micro-motion
    for _ in range(15):
        res_static = detector.compute_liveness(track_id=1, landmark=base_lm)
    assert res_static["is_live"] is False
    assert res_static["status"] == "ATTACK_SUSPECTED"
    assert res_static["score"] <= 0.20

    # 2. Dynamic live subject: 20 frames with realistic physiological jitter
    detector.reset_all()
    for _ in range(20):
        jitter = np.random.normal(0, 0.4, base_lm.shape).astype(np.float32)
        res_live = detector.compute_liveness(track_id=2, landmark=base_lm + jitter)
    assert res_live["is_live"] is True
    assert res_live["status"] == "LIVE_PERSON"
    assert res_live["score"] >= 0.65

    # 3. Single frame initial connection (no false restriction)
    detector.reset_all()
    res_initial = detector.compute_liveness(track_id=3, landmark=base_lm)
    assert res_initial["is_live"] is True
    assert res_initial["status"] == "ANALYZING"


def test_face_access_controller_enrolled_recognition(temp_db):
    """Verify out-of-the-box recognition for Ashutosh and Dhrunil."""
    controller = FaceAccessController(db_manager=temp_db)
    assert controller.is_ready is True
    assert "Ashutosh" in controller.known_names
    assert "Dhrunil" in controller.known_names

    # Test Ashutosh image
    ash_img = cv2.imread("datasets/data/ashutosh/0.jpg")
    res_ash = controller.verify_frame(ash_img, recognition_thresh=0.40)
    assert res_ash["success"] is True
    assert res_ash["primary_access"]["authorized"] is True
    assert res_ash["primary_access"]["name"] == "Ashutosh"
    assert res_ash["primary_access"]["similarity"] >= 0.85

    # Test Dhrunil image
    dhr_img = cv2.imread("datasets/data/dhrunil/0.jpg")
    res_dhr = controller.verify_frame(dhr_img, recognition_thresh=0.40)
    assert res_dhr["success"] is True
    assert res_dhr["primary_access"]["authorized"] is True
    assert res_dhr["primary_access"]["name"] == "Dhrunil"
    assert res_dhr["primary_access"]["similarity"] >= 0.85


def test_face_access_controller_unknown_classification(temp_db):
    """Verify that unregistered/unknown faces are properly classified as UNKNOWN without false restrictions."""
    controller = FaceAccessController(db_manager=temp_db)

    # Use Lam's image (enrolled in backup, but not in AUTHORIZED_SUBJECTS)
    lam_img = cv2.imread("datasets/data/lam/0.jpg")
    if lam_img is not None:
        res = controller.verify_frame(lam_img, recognition_thresh=0.40)
        assert res["success"] is True
        # If Lam is in feature.npz, it may recognize Lam, but is_authorized is False because Lam is not in AUTHORIZED_SUBJECTS
        primary = res["primary_access"]
        assert primary["authorized"] is False
        assert primary["reason"] in ("UNKNOWN_SUBJECT",)

    # Test with a synthetic face image with completely non-matching embedding
    # Mocking recognizer to return an orthogonal embedding
    original_recognizer = controller.recognizer

    class OrthoRecognizer(torch.nn.Module):
        def forward(self, x):
            # Orthogonal vector to all trained embeddings
            vec = torch.zeros(1, 512)
            vec[0, 511] = 1.0
            return vec

    controller.recognizer = OrthoRecognizer()
    ash_img = cv2.imread("datasets/data/ashutosh/0.jpg")
    res_ortho = controller.verify_frame(ash_img, recognition_thresh=0.40)
    controller.recognizer = original_recognizer

    assert res_ortho["success"] is True
    assert res_ortho["primary_access"]["authorized"] is False
    assert res_ortho["primary_access"]["name"] == "UNKNOWN"
    assert res_ortho["primary_access"]["reason"] == "UNKNOWN_SUBJECT"


def test_cosine_similarity_threshold_sweep():
    """Verify compare_encodings cosine similarity behavior across thresholds."""
    ref_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db_embs = np.array(
        [
            [1.0, 0.0, 0.0],  # similarity 1.0 (identical)
            [0.8, 0.6, 0.0],  # similarity 0.8 (same subject, different lighting)
            [0.3, 0.9, 0.0],  # similarity 0.3 (different subject)
            [0.0, 1.0, 0.0],  # similarity 0.0 (orthogonal)
        ],
        dtype=np.float32,
    )

    # Score against database
    score, idx = compare_encodings(ref_emb, db_embs)
    assert idx == 0
    assert abs(score - 1.0) < 1e-5

    # Threshold checks
    thresh = 0.40
    # Match at index 1: score 0.8 >= 0.40 -> True
    score1, _ = compare_encodings(db_embs[1], np.array([ref_emb]))
    assert score1 >= thresh

    # Match at index 2: score 0.3 < 0.40 -> False
    score2, _ = compare_encodings(db_embs[2], np.array([ref_emb]))
    assert score2 < thresh
