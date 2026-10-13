"""
Unified Face Recognition, Tracking, and Liveness Service
Integrates SCRFD detection, norm_crop alignment, ArcFace feature extraction,
BYTETracker multi-target tracking, and anti-spoofing liveness detection.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from database.db import DatabaseManager
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings
from face_tracking.tracker.byte_tracker import BYTETracker


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def enhance_low_light(img: np.ndarray) -> np.ndarray:
    """Applies CLAHE adaptive histogram equalization in LAB space for low-light frames."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class FaceService:
    """Production service orchestrating face detection, recognition, tracking, and liveness."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        detector_weights: str = "face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
        recognizer_weights: str = "face_recognition/arcface/weights/arcface_r100.pth",
        device: Optional[str] = None,
    ):
        self.db = db_manager
        self.detector_weights = detector_weights
        self.recognizer_weights = recognizer_weights

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.detector: Optional[SCRFD] = None
        self.recognizer: Optional[torch.nn.Module] = None
        self.tracker: Optional[BYTETracker] = None

        self.is_ready = False
        self.status_message = "Initializing"

        # In-memory participant embeddings cache
        self.known_names: List[str] = []
        self.known_embeddings: Optional[np.ndarray] = None
        self.participant_id_map: Dict[str, int] = {}

        # Tracking and liveness state per track ID
        self.track_history: Dict[int, Dict[str, Any]] = {}

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self._initialize_models()
        self.reload_known_faces()

    def _initialize_models(self):
        """Defensively loads SCRFD, ArcFace, and BYTETracker models."""
        try:
            # Resolve paths relative to project root if relative path is not in current working dir
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if not os.path.exists(self.detector_weights):
                alt_detector = os.path.join(base_dir, self.detector_weights)
                if os.path.exists(alt_detector):
                    self.detector_weights = alt_detector

            if not os.path.exists(self.recognizer_weights):
                alt_recognizer = os.path.join(base_dir, self.recognizer_weights)
                if os.path.exists(alt_recognizer):
                    self.recognizer_weights = alt_recognizer

            # 1. Detector
            if os.path.exists(self.detector_weights):
                self.detector = SCRFD(model_file=self.detector_weights)
                self.detector.prepare(ctx_id=-1)  # CPU inference
            else:
                self.status_message = f"Detector weight missing at {self.detector_weights}"
                return

            # 2. Recognizer
            if os.path.exists(self.recognizer_weights):
                try:
                    self.recognizer = iresnet_inference(
                        model_name="r100",
                        path=self.recognizer_weights,
                        device=str(self.device),
                    )
                except Exception as e:
                    self.status_message = f"Failed to load ArcFace weights: {e}"
                    return
            else:
                self.status_message = f"ArcFace weight missing at {self.recognizer_weights}"
                return

            # 3. Tracker
            tracker_args = {
                "track_thresh": 0.45,
                "track_buffer": 30,
                "match_thresh": 0.75,
                "min_box_area": 10,
                "aspect_ratio_thresh": 1.6,
            }
            self.tracker = BYTETracker(args=tracker_args, frame_rate=30)

            self.is_ready = True
            self.status_message = "Face Recognition Pipeline Ready"
        except Exception as e:
            self.is_ready = False
            self.status_message = f"Initialization error: {str(e)}"

    def reload_known_faces(self):
        """Loads enrolled participant embeddings from database and feature.npz multi-templates into memory."""
        participants = self.db.get_all_participants()
        names = []
        embeddings = []
        self.participant_id_map.clear()

        # 1. Map participants from SQLite database
        for p in participants:
            self.participant_id_map[p["name"]] = p["participant_id"]
            if p.get("embedding") is not None and len(p["embedding"]) > 0:
                names.append(p["name"])
                embeddings.append(p["embedding"])

        # 2. Also load multi-template gallery from datasets/face_features/feature.npz
        feat_path = "datasets/face_features/feature.npz"
        if not os.path.exists(feat_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_feat = os.path.join(base_dir, feat_path)
            if os.path.exists(alt_feat):
                feat_path = alt_feat

        if os.path.exists(feat_path):
            try:
                npz_data = np.load(feat_path, allow_pickle=True)
                npz_names = list(npz_data["images_name"])
                npz_embs = npz_data["images_emb"]
                for n, e in zip(npz_names, npz_embs):
                    c_name = str(n).strip().capitalize()
                    names.append(c_name)
                    embeddings.append(e)
                    if c_name not in self.participant_id_map:
                        for p in participants:
                            if p["name"].lower() == c_name.lower():
                                self.participant_id_map[c_name] = p["participant_id"]
                                break
            except Exception as e:
                logger.warning(f"Could not load multi-templates from {feat_path}: {e}")

        self.known_names = names
        if embeddings:
            self.known_embeddings = np.array(embeddings)
        else:
            self.known_embeddings = None

    @torch.no_grad()
    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extracts and L2-normalizes 512-dim ArcFace embedding from aligned face."""
        if self.recognizer is None:
            raise RuntimeError("Recognizer model is not loaded")

        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        tensor_img = self.preprocess(rgb_face).unsqueeze(0).to(self.device)
        raw_emb = self.recognizer(tensor_img).cpu().numpy()[0]
        norm = np.linalg.norm(raw_emb)
        if norm > 0:
            norm_emb = raw_emb / norm
        else:
            norm_emb = raw_emb
        return norm_emb

    @staticmethod
    def estimate_head_pose(landmark: np.ndarray) -> Dict[str, Any]:
        """
        Estimates head pose (Yaw, Pitch, Roll in degrees) from 5 facial landmarks:
        [left_eye, right_eye, nose, left_mouth, right_mouth].
        """
        left_eye, right_eye, nose, left_mouth, right_mouth = [np.array(pt, dtype=np.float32) for pt in landmark]

        # 1. Roll: Angle between the two eyes relative to horizontal
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Inter-ocular distance
        iod = float(np.linalg.norm(right_eye - left_eye))
        eye_mid = (left_eye + right_eye) / 2.0

        # 2. Yaw: Nose horizontal displacement from eye midpoint normalized by IOD
        yaw_ratio = float((nose[0] - eye_mid[0]) / (iod + 1e-6))
        # Frontal: ~0.0. 30 deg turn corresponds to ratio ~0.25 - 0.35
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
        - Anatomical vertical ordering: nose positioned between eyes and mouth along face axis
        """
        left_eye, right_eye, nose, left_mouth, right_mouth = [np.array(pt, dtype=np.float32) for pt in landmark]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        mouth_dist = float(np.linalg.norm(right_mouth - left_mouth))
        eye_center = 0.5 * (left_eye + right_eye)
        mouth_center = 0.5 * (left_mouth + right_mouth)
        nose_mouth_dist = float(np.linalg.norm(nose - mouth_center))

        emr = float(nose_mouth_dist / (eye_dist + 1e-6))
        mer = float(mouth_dist / (eye_dist + 1e-6))

        # Anatomical vertical ordering check:
        # Vector from eyes center to mouth center defines the facial longitudinal axis
        v_face = mouth_center - eye_center
        v_face_len = float(np.linalg.norm(v_face))
        if v_face_len > 1e-6:
            u_face = v_face / v_face_len
            t_nose = float(np.dot(nose - eye_center, u_face) / v_face_len)
            ordered_vertical = (-0.10 <= t_nose <= 1.15)
        else:
            ordered_vertical = True

        # Biological plausibility boundaries for human facial geometry (accommodating pitch/yaw & perspective)
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

        # Keep rolling window of 30 frames (~1 sec at 30fps)
        if len(hist["timestamps"]) > 30:
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
            # Distorted geometry sustained over multiple frames (e.g. paper cutout, folded mask, extreme non-face artifact)
            score = 0.20
            status = "UNNATURAL_GEOMETRY"
            is_live = False
        elif history_len >= 12 and internal_dynamic_var < 1.0e-6:
            # Static photo attack: natural human physiological tremor/micro-expression is absent
            score = 0.15
            status = "ATTACK_SUSPECTED"
            is_live = False
        elif pose["is_extreme"]:
            # Face turned past reliable recognition angles
            score = 0.60
            status = "EXTREME_HEAD_ANGLE"
            is_live = True
        elif history_len < 6:
            # Initial accumulation phase
            score = 0.55
            status = "ANALYZING"
            is_live = True
        else:
            # Verified dynamic live subject with natural micro-motions
            # Base score scaled smoothly by physiological variance band
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

    def process_frame(
        self,
        frame: np.ndarray,
        recognition_thresh: float = 0.40,
    ) -> Dict[str, Any]:
        """
        Executes full pipeline on a video frame:
        Detection (SCRFD) -> Tracking (BYTETracker) -> Low-light check/enhance ->
        Norm-crop Alignment -> Feature Extraction (ArcFace) -> Recognition -> Liveness.
        """
        if not self.is_ready or self.detector is None or frame is None or frame.size == 0:
            return {"annotated_frame": frame, "detections": [], "status": self.status_message}

        h, w, _ = frame.shape
        bboxes, landmarks = self.detector.detect(image=frame)

        # Global frame lighting check
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_luma = float(np.mean(gray_frame))
        is_global_low_light = frame_luma < 45.0

        # Multi-target tracking update with BYTETracker
        tracked_targets = []
        if self.tracker is not None and len(bboxes) > 0:
            try:
                tracked_targets = self.tracker.update(bboxes, [h, w], (h, w))
            except Exception:
                tracked_targets = []

        detections = []
        annotated_frame = frame.copy()

        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            x1, y1, x2, y2, det_score = [int(v) for v in bbox[:4]] + [float(bbox[4])]
            landmark = landmarks[idx]

            # Bounding box sanity checks
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # Associate detection with persistent BYTETracker track ID
            track_id = idx + 1
            best_iou = 0.0
            for t in tracked_targets:
                iou = compute_iou([x1, y1, x2, y2], t.tlbr)
                if iou > best_iou and iou >= 0.30:
                    best_iou = iou
                    track_id = int(t.track_id)

            # Local face ROI illumination check
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                face_luma = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)))
                is_low_light = face_luma < 45.0 or is_global_low_light
            else:
                is_low_light = is_global_low_light

            # Norm_crop alignment with adaptive low-light enhancement if necessary
            aligned_face = norm_crop(img=frame, landmark=landmark)
            if is_low_light:
                aligned_face = enhance_low_light(aligned_face)

            # Anti-spoofing and liveness evaluation
            liveness_info = self.compute_liveness(track_id=track_id, landmark=landmark, bbox=[x1, y1, x2, y2])

            # Recognition matching with ArcFace
            recognized_name = "UNKNOWN"
            similarity_score = 0.0
            participant_id = None
            is_known = False

            if self.recognizer is not None and self.known_embeddings is not None and len(self.known_embeddings) > 0:
                query_emb = self.extract_embedding(aligned_face)
                score, match_idx = compare_encodings(query_emb, self.known_embeddings)
                similarity_score = float(score)
                if similarity_score >= recognition_thresh:
                    recognized_name = self.known_names[match_idx]
                    participant_id = self.participant_id_map.get(recognized_name)
                    is_known = True

            det_info = {
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "det_score": round(det_score, 2),
                "name": recognized_name,
                "similarity": round(similarity_score, 2),
                "is_known": is_known,
                "participant_id": participant_id,
                "liveness": liveness_info,
                "head_pose": liveness_info.get("pose"),
                "low_light": is_low_light,
            }
            detections.append(det_info)

            # Visual Annotation
            if not liveness_info["is_live"]:
                box_color = (0, 0, 255)  # Red for attack / spoof suspected
            elif liveness_info.get("pose", {}).get("is_extreme"):
                box_color = (0, 255, 255)  # Yellow for extreme head angle
            elif is_known:
                box_color = (0, 255, 0)  # Green for recognized live person
            else:
                box_color = (0, 165, 255)  # Orange for unknown live person

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw landmark points
            for pt in landmark:
                cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

            status_tag = liveness_info["status"]
            if is_low_light:
                status_tag += " [LOW-LIGHT]"

            label = f"ID:{track_id} {recognized_name} ({similarity_score:.2f}) | {status_tag}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                box_color,
                2,
            )

        return {
            "annotated_frame": annotated_frame,
            "detections": detections,
            "status": "OK",
        }

    def enroll_participant(self, name: str, frame: np.ndarray, notes: str = "") -> Dict[str, Any]:
        """
        Enrolls a new participant directly from a camera frame into SQLite database.
        Handles corner cases: multiple faces, extreme head angles, low-light frames, blur, and size.
        """
        if not self.is_ready or self.detector is None or self.recognizer is None:
            return {"success": False, "message": "Face recognition models not ready"}

        if frame is None or frame.size == 0:
            return {"success": False, "message": "Invalid or empty enrollment image"}

        h, w = frame.shape[:2]
        bboxes, landmarks = self.detector.detect(image=frame)

        # Corner Case 1: Face presence and multiple faces check
        if len(bboxes) == 0:
            return {
                "success": False,
                "message": "No face detected in enrollment image. Please face the camera directly.",
            }

        if len(bboxes) > 1:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            max_area = max(areas)
            # Check if there are multiple prominent faces
            prominent_faces = [a for a in areas if a > 0.30 * max_area]
            if len(prominent_faces) > 1:
                return {
                    "success": False,
                    "message": f"Multiple faces detected ({len(bboxes)} found). Please ensure only the participant is in frame.",
                }
            # Pick primary dominant face
            primary_idx = int(np.argmax(areas))
            bbox = bboxes[primary_idx]
            landmark = landmarks[primary_idx]
        else:
            bbox = bboxes[0]
            landmark = landmarks[0]

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_w, box_h = x2 - x1, y2 - y1

        # Corner Case 2: Face resolution and minimum area
        if box_w < 60 or box_h < 60:
            return {
                "success": False,
                "message": f"Face resolution too low ({box_w}x{box_h} px). Please move closer to the camera.",
            }

        # Corner Case 3: Extreme head angles during enrollment
        pose = self.estimate_head_pose(landmark)
        if abs(pose["yaw"]) > 25.0 or abs(pose["pitch"]) > 25.0 or abs(pose["roll"]) > 25.0:
            return {
                "success": False,
                "message": f"Extreme head angle detected (Yaw: {pose['yaw']}°, Pitch: {pose['pitch']}°, Roll: {pose['roll']}°). Please face camera directly.",
            }

        # Corner Case 4: Illumination and sharpness checks
        face_roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        luma = float(np.mean(gray_roi))
        blur_score = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

        if luma < 30.0:
            return {
                "success": False,
                "message": f"Lighting is too dim (Luminance: {round(luma, 1)}/255). Please improve lighting.",
            }

        if blur_score < 30.0:
            return {
                "success": False,
                "message": f"Image is blurry (Sharpness: {round(blur_score, 1)}). Please hold steady for enrollment.",
            }

        # Adaptive enhancement for slightly low-light frames (30 <= luma < 55)
        enroll_frame = frame
        if luma < 55.0:
            enroll_frame = enhance_low_light(frame)

        aligned_face = norm_crop(img=enroll_frame, landmark=landmark)
        embedding = self.extract_embedding(aligned_face)

        participant_id = self.db.add_participant(name=name, embedding=embedding, notes=notes)
        self.reload_known_faces()

        return {
            "success": True,
            "message": f"Successfully enrolled '{name}'",
            "participant_id": participant_id,
            "quality": {
                "luminance": round(luma, 1),
                "sharpness": round(blur_score, 1),
                "pose": pose,
            },
        }
