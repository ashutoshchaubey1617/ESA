"""
Unit tests for Face Recognition Integration, Alignment, Tracking, Liveness, and Corner Cases.
"""

import os
import tempfile
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from database.db import DatabaseManager
from face_alignment.alignment import norm_crop
from face_pipeline.face_service import (
    FaceService,
    compute_iou,
    enhance_low_light,
)
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker


@pytest.fixture
def temp_db():
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    db = DatabaseManager(temp_file.name)
    yield db
    try:
        os.remove(temp_file.name)
    except Exception:
        pass


def test_compare_encodings_cosine_similarity():
    # Identical vectors should have similarity ~1.0
    v1 = np.array([1.0, 0.0, 0.0])
    db_embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    score, match_idx = compare_encodings(v1, db_embs)
    assert match_idx == 0
    assert abs(score - 1.0) < 1e-5

    # Orthogonal vectors should have similarity ~0.0
    v2 = np.array([0.0, 0.0, 1.0])
    score_ortho, _ = compare_encodings(v2, db_embs)
    assert abs(score_ortho) < 1e-5

    # Safe empty encodings check (bugfix verification)
    score_empty, idx_empty = compare_encodings(v1, [])
    assert score_empty[0] == 0.0
    assert idx_empty == 0


def test_norm_crop_alignment():
    # 5 standard reference points: left eye, right eye, nose, left mouth, right mouth
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_landmark = np.array(
        [[100, 100], [200, 100], [150, 150], [110, 220], [190, 220]], dtype=np.float32
    )
    cropped = norm_crop(dummy_img, dummy_landmark, image_size=112)
    assert cropped.shape == (112, 112, 3)


def test_face_service_initialization(temp_db):
    service = FaceService(temp_db)
    # Both scrfd and arcface weights are on disk
    assert service.is_ready is True
    assert service.detector is not None
    assert service.recognizer is not None
    assert service.tracker is not None


def test_compute_iou():
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    assert abs(compute_iou(box1, box2) - 1.0) < 1e-5

    box3 = [100, 100, 150, 150]
    assert compute_iou(box1, box3) == 0.0

    box4 = [30, 10, 70, 50]
    iou = compute_iou(box1, box4)
    assert 0.0 < iou < 1.0


def test_head_pose_estimation(temp_db):
    service = FaceService(temp_db)

    # Frontal symmetric face
    frontal_landmark = np.array(
        [[100, 100], [200, 100], [150, 150], [115, 200], [185, 200]], dtype=np.float32
    )
    pose_frontal = service.estimate_head_pose(frontal_landmark)
    assert abs(pose_frontal["roll"]) < 2.0
    assert abs(pose_frontal["yaw"]) < 5.0
    assert pose_frontal["is_extreme"] is False

    # Turned face (extreme yaw: nose shifted heavily to the right)
    turned_landmark = np.array(
        [[100, 100], [200, 100], [190, 150], [120, 200], [190, 200]], dtype=np.float32
    )
    pose_turned = service.estimate_head_pose(turned_landmark)
    assert pose_turned["yaw"] > 30.0
    assert pose_turned["is_extreme"] is True

    # Tilted head (extreme roll: 45 degree tilt)
    tilted_landmark = np.array(
        [[100, 100], [170, 170], [120, 150], [90, 190], [150, 210]], dtype=np.float32
    )
    pose_tilted = service.estimate_head_pose(tilted_landmark)
    assert abs(pose_tilted["roll"]) > 32.0
    assert pose_tilted["is_extreme"] is True


def test_check_geometric_proportions(temp_db):
    service = FaceService(temp_db)

    # Normal proportions
    normal_landmark = np.array(
        [[100, 100], [200, 100], [150, 150], [115, 200], [185, 200]], dtype=np.float32
    )
    geom_normal = service.check_geometric_proportions(normal_landmark)
    assert geom_normal["is_plausible"] is True

    # Severely warped geometry (mouth placed above the eyes)
    warped_landmark = np.array(
        [[100, 100], [200, 100], [150, 150], [115, 50], [185, 50]], dtype=np.float32
    )
    geom_warped = service.check_geometric_proportions(warped_landmark)
    assert geom_warped["is_plausible"] is False


def test_liveness_detection_dynamics_live_person(temp_db):
    service = FaceService(temp_db)
    landmark_base = np.array([[100, 100], [200, 100], [150, 150], [115, 200], [185, 200]], dtype=np.float32)

    # Feed 20 frames with realistic human physiological micro-motion
    for _ in range(20):
        jitter = np.random.normal(0, 0.4, landmark_base.shape).astype(np.float32)
        live_res = service.compute_liveness(track_id=10, landmark=landmark_base + jitter)

    assert live_res["is_live"] is True
    assert live_res["status"] == "LIVE_PERSON"
    assert live_res["score"] >= 0.65


def test_liveness_static_photo_attack(temp_db):
    service = FaceService(temp_db)
    landmark_static = np.array([[100, 100], [200, 100], [150, 150], [115, 200], [185, 200]], dtype=np.float32)

    # Feed 20 completely static frames (zero variance - printed photo presentation attack)
    for _ in range(20):
        res = service.compute_liveness(track_id=20, landmark=landmark_static)

    assert res["is_live"] is False
    assert res["status"] == "ATTACK_SUSPECTED"
    assert res["score"] <= 0.25


def test_low_light_enhancement():
    # Create dark image (average intensity ~15)
    dark_img = np.full((100, 100, 3), 15, dtype=np.uint8)
    enhanced = enhance_low_light(dark_img)

    assert enhanced.shape == dark_img.shape
    assert np.mean(enhanced) > np.mean(dark_img)


def test_bytetracker_tracking_across_frames():
    tracker_args = {
        "track_thresh": 0.45,
        "track_buffer": 30,
        "match_thresh": 0.75,
        "min_box_area": 10,
        "aspect_ratio_thresh": 1.6,
    }
    tracker = BYTETracker(args=tracker_args, frame_rate=30)

    # Frame 1
    det1 = np.array([[50, 50, 150, 150, 0.95]], dtype=np.float32)
    tracks1 = tracker.update(det1, [480, 640], (480, 640))
    assert len(tracks1) == 1
    t1_id = tracks1[0].track_id

    # Frame 2 with small translation
    det2 = np.array([[52, 51, 152, 151, 0.93]], dtype=np.float32)
    tracks2 = tracker.update(det2, [480, 640], (480, 640))
    assert len(tracks2) == 1
    assert tracks2[0].track_id == t1_id


def test_enrollment_corner_cases_no_face(temp_db):
    service = FaceService(temp_db)
    # Blank black image has no face
    blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = service.enroll_participant(name="Alice", frame=blank_img)
    assert res["success"] is False
    assert "No face detected" in res["message"]


def test_enrollment_corner_cases_multiple_faces(temp_db):
    service = FaceService(temp_db)
    dummy_img = np.ones((400, 400, 3), dtype=np.uint8) * 128

    # Mock detector returning two equal-sized prominent faces
    service.detector.detect = MagicMock(return_value=(
        np.array([[50, 50, 150, 150, 0.9], [220, 50, 320, 150, 0.9]]),
        np.array([
            [[80, 80], [120, 80], [100, 105], [85, 125], [115, 125]],
            [[250, 80], [290, 80], [270, 105], [255, 125], [285, 125]],
        ])
    ))

    res = service.enroll_participant(name="Bob", frame=dummy_img)
    assert res["success"] is False
    assert "Multiple faces detected" in res["message"]


def test_enrollment_corner_cases_extreme_angle(temp_db):
    service = FaceService(temp_db)
    dummy_img = np.ones((400, 400, 3), dtype=np.uint8) * 128

    # Mock detector returning one face with extreme yaw (nose way off center)
    service.detector.detect = MagicMock(return_value=(
        np.array([[50, 50, 250, 250, 0.95]]),
        np.array([
            [[80, 100], [220, 100], [215, 160], [100, 210], [200, 210]]
        ])
    ))

    res = service.enroll_participant(name="Charlie", frame=dummy_img)
    assert res["success"] is False
    assert "Extreme head angle" in res["message"]


def test_enrollment_corner_cases_low_resolution(temp_db):
    service = FaceService(temp_db)
    dummy_img = np.ones((400, 400, 3), dtype=np.uint8) * 128

    # Tiny box 30x30
    service.detector.detect = MagicMock(return_value=(
        np.array([[50, 50, 80, 80, 0.95]]),
        np.array([[[60, 60], [70, 60], [65, 68], [62, 74], [68, 74]]])
    ))

    res = service.enroll_participant(name="David", frame=dummy_img)
    assert res["success"] is False
    assert "resolution too low" in res["message"].lower()


def test_enrollment_corner_cases_dark_image(temp_db):
    service = FaceService(temp_db)
    # Extremely dark frame
    dark_img = np.full((300, 300, 3), 10, dtype=np.uint8)

    service.detector.detect = MagicMock(return_value=(
        np.array([[50, 50, 200, 200, 0.95]]),
        np.array([[[90, 90], [160, 90], [125, 130], [100, 165], [150, 165]]])
    ))

    res = service.enroll_participant(name="Eve", frame=dark_img)
    assert res["success"] is False
    assert "too dim" in res["message"].lower()


def test_enrollment_success_with_quality_metrics(temp_db):
    service = FaceService(temp_db)
    # Normal well-lit face frame
    bright_img = np.ones((300, 300, 3), dtype=np.uint8) * 140
    # Add subtle pattern so sharpness/Laplacian variance is high (> 30)
    bright_img[::2, ::2] = 200

    service.detector.detect = MagicMock(return_value=(
        np.array([[50, 50, 220, 220, 0.95]]),
        np.array([[[90, 90], [170, 90], [130, 135], [100, 175], [160, 175]]])
    ))

    res = service.enroll_participant(name="Dr. Smith", frame=bright_img, notes="Cardiology")
    assert res["success"] is True
    assert "Dr. Smith" in res["message"]
    assert "quality" in res
    assert res["quality"]["luminance"] > 50
    assert "pose" in res["quality"]
    assert "Dr. Smith" in service.known_names
