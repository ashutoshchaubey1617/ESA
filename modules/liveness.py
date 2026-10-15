"""
Anti-Spoofing and Liveness Detection Module.
Adheres to the vectornguyen76 face-recognition pipeline standards.

Provides:
- 5-point facial landmark geometric plausibility evaluation
- 3D head pose estimation (Yaw, Pitch, Roll)
- Multi-frame micro-motion dynamic variance tracking
- Presentation attack detection (static photograph and warped mask rejection)
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np


class LivenessDetector:
    """
    Evaluates face anti-spoofing and liveness using landmark geometry,
    head pose dynamics, and physiological micro-motions across rolling frames.
    """

    def __init__(self, history_window: int = 30):
        self.history_window = history_window
        self.track_history: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def estimate_head_pose(landmark: np.ndarray) -> Dict[str, Any]:
        """
        Estimates head pose (Yaw, Pitch, Roll in degrees) from 5 facial landmarks:
        [left_eye, right_eye, nose, left_mouth, right_mouth] conforming to SCRFD standard.
        """
        left_eye, right_eye, nose, left_mouth, right_mouth = [
            np.array(pt, dtype=np.float32) for pt in landmark
        ]

        # 1. Roll: Angle between the two eyes relative to horizontal
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Inter-ocular distance (IOD)
        iod = float(np.linalg.norm(right_eye - left_eye))
        eye_mid = (left_eye + right_eye) / 2.0

        # 2. Yaw: Nose horizontal displacement from eye midpoint normalized by IOD
        yaw_ratio = float((nose[0] - eye_mid[0]) / (iod + 1e-6))
        yaw_deg = float(np.clip(yaw_ratio * 95.0, -90.0, 90.0))

        # 3. Pitch: Ratio of nose-to-eyes vs nose-to-mouth vertical distances
        mouth_mid = (left_mouth + right_mouth) / 2.0
        dist_ne = float(nose[1] - eye_mid[1])
        dist_nm = float(mouth_mid[1] - nose[1])
        pitch_ratio = float((dist_ne - dist_nm) / (iod + 1e-6))
        pitch_deg = float(np.clip(pitch_ratio * 80.0, -60.0, 60.0))

        is_extreme = abs(yaw_deg) > 32.0 or abs(pitch_deg) > 28.0 or abs(roll_deg) > 32.0

        return {
            "yaw": round(yaw_deg, 1),
            "pitch": round(pitch_deg, 1),
            "roll": round(roll_deg, 1),
            "is_extreme": is_extreme,
        }

    @staticmethod
    def check_geometric_proportions(landmark: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates biological facial geometry proportions:
        - Eye-to-Mouth Ratio (EMR): nose_mouth_dist / eye_dist
        - Mouth-to-Eye Ratio (MER): mouth_width / eye_dist
        - Anatomical vertical ordering: nose positioned between eyes and mouth
        """
        left_eye, right_eye, nose, left_mouth, right_mouth = [
            np.array(pt, dtype=np.float32) for pt in landmark
        ]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        mouth_dist = float(np.linalg.norm(right_mouth - left_mouth))
        eye_center = 0.5 * (left_eye + right_eye)
        mouth_center = 0.5 * (left_mouth + right_mouth)
        nose_mouth_dist = float(np.linalg.norm(nose - mouth_center))

        emr = float(nose_mouth_dist / (eye_dist + 1e-6))
        mer = float(mouth_dist / (eye_dist + 1e-6))

        # Anatomical vertical ordering check:
        v_face = mouth_center - eye_center
        v_face_len = float(np.linalg.norm(v_face))
        if v_face_len > 1e-6:
            u_face = v_face / v_face_len
            t_nose = float(np.dot(nose - eye_center, u_face) / v_face_len)
            ordered_vertical = (-0.10 <= t_nose <= 1.15)
        else:
            ordered_vertical = True

        emr_plausible = (0.10 <= emr <= 4.2)
        mer_plausible = (0.15 <= mer <= 2.8)
        is_plausible = emr_plausible and mer_plausible and ordered_vertical

        return {
            "eye_mouth_ratio": round(emr, 3),
            "mouth_eye_ratio": round(mer, 3),
            "is_plausible": is_plausible,
        }

    def compute_liveness(
        self,
        track_id: int,
        landmark: np.ndarray,
        bbox: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates anti-spoofing liveness metrics using landmark geometry,
        head pose angles, and micro-motion dynamics across rolling video frames.
        """
        now = time.time()
        geom = self.check_geometric_proportions(landmark)
        pose = self.estimate_head_pose(landmark)
        emr = geom["eye_mouth_ratio"]
        mer = geom["mouth_eye_ratio"]

        if track_id not in self.track_history:
            self.track_history[track_id] = {
                "timestamps": [now],
                "landmarks": [landmark],
                "bboxes": [bbox] if bbox else [],
                "emr_history": [emr],
                "mer_history": [mer],
                "liveness_score": 0.5,
                "status": "ANALYZING",
            }
            return {
                "score": 0.5,
                "status": "ANALYZING",
                "is_live": True,
                "pose": pose,
                "metrics": {
                    "eye_mouth_ratio": emr,
                    "mouth_eye_ratio": mer,
                    "ratio_variance": 0.0,
                    "yaw": pose["yaw"],
                    "pitch": pose["pitch"],
                    "roll": pose["roll"],
                },
            }

        hist = self.track_history[track_id]
        hist["timestamps"].append(now)
        hist["landmarks"].append(landmark)
        hist["emr_history"].append(emr)
        hist["mer_history"].append(mer)
        if bbox:
            hist["bboxes"].append(bbox)

        # Retain rolling window
        if len(hist["timestamps"]) > self.history_window:
            hist["timestamps"].pop(0)
            hist["landmarks"].pop(0)
            hist["emr_history"].pop(0)
            hist["mer_history"].pop(0)
            if hist["bboxes"]:
                hist["bboxes"].pop(0)

        history_len = len(hist["emr_history"])
        emr_var = float(np.var(hist["emr_history"])) if history_len >= 6 else 0.001
        mer_var = float(np.var(hist["mer_history"])) if history_len >= 6 else 0.001
        internal_dynamic_var = 0.5 * (emr_var + mer_var)

        # Micro-motion analysis & Anti-spoofing decision tree
        if not geom["is_plausible"] and history_len >= 8:
            # Distorted geometry sustained over multiple frames (e.g., photo cutout, paper mask)
            score = 0.20
            status = "UNNATURAL_GEOMETRY"
            is_live = False
        elif history_len >= 12 and internal_dynamic_var < 1.0e-6:
            # Static photo presentation attack: zero micro-tremor over >= 12 consecutive frames
            score = 0.15
            status = "ATTACK_SUSPECTED"
            is_live = False
        elif pose["is_extreme"]:
            # Extreme head angle (valid face turned away)
            score = 0.60
            status = "EXTREME_HEAD_ANGLE"
            is_live = True
        elif history_len < 6:
            # Initial accumulation phase
            score = 0.55
            status = "ANALYZING"
            is_live = True
        else:
            # Verified dynamic live subject with natural physiological tremor
            scaled_score = 0.70 + 40.0 * float(np.clip(internal_dynamic_var, 0.0, 0.0075))
            score = min(0.99, max(0.65, scaled_score))
            status = "LIVE_PERSON"
            is_live = True

        hist["liveness_score"] = score
        hist["status"] = status

        return {
            "score": round(float(score), 2),
            "status": status,
            "is_live": is_live,
            "pose": pose,
            "metrics": {
                "eye_mouth_ratio": emr,
                "mouth_eye_ratio": mer,
                "ratio_variance": float(internal_dynamic_var),
                "yaw": pose["yaw"],
                "pitch": pose["pitch"],
                "roll": pose["roll"],
            },
        }

    def reset_track(self, track_id: int):
        """Clears tracking history for a specific track ID."""
        if track_id in self.track_history:
            del self.track_history[track_id]

    def reset_all(self):
        """Clears all tracking histories."""
        self.track_history.clear()


# Module-level convenience functions matching FaceService interface
_default_detector = LivenessDetector()
estimate_head_pose = _default_detector.estimate_head_pose
check_geometric_proportions = _default_detector.check_geometric_proportions
compute_liveness = _default_detector.compute_liveness
