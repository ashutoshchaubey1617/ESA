"""
Face Access Control and Verification Module.
Adheres strictly to the vectornguyen76 face-recognition module standards:
- Detection: SCRFD ONNX (multi-scale 320x320 / 640x640 / 128x128)
- Alignment: norm_crop 5-point affine transformation to standard 112x112 ArcFace format
- Feature Extraction: ArcFace iResNet-100 generating L2-normalized 512-dim embeddings
- Matching: Cosine similarity via compare_encodings (dot product on unit vectors)
- Tracking: BYTETracker multi-target tracking with Kalman filtering
- Anti-Spoofing: Landmark dynamics & micro-motion analysis via LivenessDetector
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torchvision import transforms

from database.db import DatabaseManager
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from modules.liveness import LivenessDetector


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


class FaceAccessController:
    """
    Modular Face Access Controller managing detection, tracking, liveness,
    biometric matching, and enrollment dataset verification.
    """

    AUTHORIZED_SUBJECTS = {"ashutosh", "dhrunil"}
    DEFAULT_COSINE_THRESHOLD = 0.40

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        detector_weights: str = "face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
        recognizer_weights: str = "face_recognition/arcface/weights/arcface_r100.pth",
        features_npz_path: str = "datasets/face_features/feature.npz",
        device: Optional[str] = None,
        recognition_thresh: float = DEFAULT_COSINE_THRESHOLD,
    ):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db = db_manager if db_manager is not None else DatabaseManager()

        self.detector_weights = self._resolve_path(detector_weights)
        self.recognizer_weights = self._resolve_path(recognizer_weights)
        self.features_npz_path = self._resolve_path(features_npz_path)
        self.recognition_thresh = recognition_thresh

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.detector: Optional[SCRFD] = None
        self.recognizer: Optional[torch.nn.Module] = None
        self.tracker: Optional[BYTETracker] = None
        self.liveness_detector = LivenessDetector()

        # In-memory participant embeddings cache
        self.known_names: List[str] = []
        self.known_embeddings: Optional[np.ndarray] = None
        self.participant_id_map: Dict[str, int] = {}

        self.is_ready = False
        self.status_message = "Initializing"

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self._initialize_pipeline()
        self.reload_enrollment_dataset()

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        alt = os.path.join(self.base_dir, path)
        if os.path.exists(alt):
            return alt
        return path

    def _initialize_pipeline(self):
        """Loads SCRFD detector, ArcFace recognizer, and BYTETracker."""
        try:
            # 1. Detector (SCRFD ONNX)
            if os.path.exists(self.detector_weights):
                self.detector = SCRFD(model_file=self.detector_weights)
                self.detector.prepare(ctx_id=-1)  # CPU execution
            else:
                self.status_message = f"Missing SCRFD weights: {self.detector_weights}"
                return

            # 2. Recognizer (ArcFace iResNet-100 PyTorch)
            if os.path.exists(self.recognizer_weights):
                self.recognizer = iresnet_inference(
                    model_name="r100",
                    path=self.recognizer_weights,
                    device=str(self.device),
                )
            else:
                self.status_message = f"Missing ArcFace weights: {self.recognizer_weights}"
                return

            # 3. Tracker (BYTETracker)
            tracker_args = {
                "track_thresh": 0.45,
                "track_buffer": 30,
                "match_thresh": 0.75,
                "min_box_area": 10,
                "aspect_ratio_thresh": 1.6,
            }
            self.tracker = BYTETracker(args=tracker_args, frame_rate=30)

            self.is_ready = True
            self.status_message = "Face Access Controller Pipeline Ready"
        except Exception as e:
            self.is_ready = False
            self.status_message = f"Initialization error: {str(e)}"

    def reload_enrollment_dataset(self):
        """
        Loads enrolled participant embeddings from:
        1. SQLite health_monitor.db (Ashutosh & Dhrunil records)
        2. datasets/face_features/feature.npz (multi-template gallery)
        """
        participants = self.db.get_all_participants()
        names = []
        embeddings = []
        self.participant_id_map.clear()

        # 1. SQLite database participants
        for p in participants:
            c_name = p["name"].strip().capitalize()
            self.participant_id_map[c_name] = p["participant_id"]
            if p.get("embedding") is not None and len(p["embedding"]) > 0:
                names.append(c_name)
                embeddings.append(p["embedding"])

        # 2. Multi-template gallery from feature.npz
        if os.path.exists(self.features_npz_path):
            try:
                npz_names, npz_embs = read_features(
                    self.features_npz_path.replace(".npz", "")
                )
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
                pass

        self.known_names = names
        if embeddings:
            self.known_embeddings = np.array(embeddings)
        else:
            self.known_embeddings = None

    @torch.no_grad()
    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Conforms to vectornguyen76 standard:
        112x112 aligned RGB image -> tensor normalization -> iResNet-100 -> L2 unit norm.
        """
        if self.recognizer is None:
            raise RuntimeError("ArcFace recognizer model not initialized")

        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        tensor_img = self.preprocess(rgb_face).unsqueeze(0).to(self.device)
        raw_emb = self.recognizer(tensor_img).cpu().numpy()[0]
        norm = np.linalg.norm(raw_emb)
        return raw_emb / norm if norm > 0 else raw_emb

    def verify_frame(
        self,
        frame: np.ndarray,
        recognition_thresh: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete verification flow adhering to vectornguyen76 module standards:
        1. Multi-scale SCRFD face detection
        2. BYTETracker multi-target tracking
        3. 5-point landmark norm_crop alignment
        4. ArcFace iResNet-100 feature extraction
        5. Cosine similarity matching (compare_encodings)
        6. Liveness anti-spoofing evaluation
        7. Authorized / Unknown classification without false restrictions
        """
        thresh = recognition_thresh if recognition_thresh is not None else self.recognition_thresh

        if not self.is_ready or self.detector is None or frame is None or frame.size == 0:
            return {
                "success": False,
                "status": self.status_message,
                "detections": [],
                "primary_access": {"authorized": False, "reason": "NOT_READY"},
            }

        h, w, _ = frame.shape

        # Multi-scale detection: try 320x320 first, fallback to 640x640 or 128x128
        bboxes, landmarks = self.detector.detect(image=frame, thresh=0.35, input_size=(320, 320))
        if len(bboxes) == 0:
            bboxes, landmarks = self.detector.detect(image=frame, thresh=0.35, input_size=(640, 640))
        if len(bboxes) == 0:
            bboxes, landmarks = self.detector.detect(image=frame, thresh=0.35, input_size=(128, 128))

        if len(bboxes) == 0:
            return {
                "success": False,
                "status": "NO_FACE_DETECTED",
                "detections": [],
                "primary_access": {
                    "authorized": False,
                    "reason": "NO_FACE",
                    "message": "No face detected in camera frame.",
                },
            }

        # Global illumination check
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_global_low_light = float(np.mean(gray_frame)) < 45.0

        # BYTETracker tracking update
        tracked_targets = []
        if self.tracker is not None:
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

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # Associate track ID
            track_id = idx + 1
            best_iou = 0.0
            for t in tracked_targets:
                iou = compute_iou([x1, y1, x2, y2], t.tlbr)
                if iou > best_iou and iou >= 0.30:
                    best_iou = iou
                    track_id = int(t.track_id)

            # Local illumination
            face_roi = frame[y1:y2, x1:x2]
            face_luma = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))) if face_roi.size > 0 else 0.0
            is_low_light = face_luma < 45.0 or is_global_low_light

            # Vectornguyen76 norm_crop 5-point alignment
            aligned_face = norm_crop(img=frame, landmark=landmark)
            if is_low_light:
                aligned_face = enhance_low_light(aligned_face)

            # Liveness evaluation
            liveness_info = self.liveness_detector.compute_liveness(
                track_id=track_id, landmark=landmark, bbox=[x1, y1, x2, y2]
            )

            # Cosine similarity matching with ArcFace
            recognized_name = "UNKNOWN"
            similarity_score = 0.0
            participant_id = None
            is_known = False

            if self.recognizer is not None and self.known_embeddings is not None and len(self.known_embeddings) > 0:
                query_emb = self.extract_embedding(aligned_face)
                score, match_idx = compare_encodings(query_emb, self.known_embeddings)
                similarity_score = float(score)
                if similarity_score >= thresh:
                    recognized_name = self.known_names[match_idx]
                    participant_id = self.participant_id_map.get(recognized_name)
                    is_known = True

            is_authorized = (
                is_known
                and recognized_name.lower() in self.AUTHORIZED_SUBJECTS
                and liveness_info["is_live"]
            )

            det_info = {
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "det_score": round(det_score, 2),
                "name": recognized_name,
                "similarity": round(similarity_score, 2),
                "is_known": is_known,
                "is_authorized": is_authorized,
                "participant_id": participant_id,
                "liveness": liveness_info,
                "head_pose": liveness_info.get("pose"),
                "low_light": is_low_light,
                "landmarks": landmark.tolist() if hasattr(landmark, "tolist") else landmark,
            }
            detections.append(det_info)

            # Annotations: Green for Authorized, Orange for Unknown, Red for Attack, Yellow for Extreme Angle
            if not liveness_info["is_live"]:
                box_color = (0, 0, 255)
            elif liveness_info.get("pose", {}).get("is_extreme"):
                box_color = (0, 255, 255)
            elif is_authorized:
                box_color = (0, 255, 0)
            else:
                box_color = (0, 165, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            for pt in landmark:
                cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

            status_tag = liveness_info["status"]
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

        # Primary access decision logic (without false restrictions)
        primary = detections[0]
        liveness_status = primary["liveness"]["status"]

        if liveness_status in ("ATTACK_SUSPECTED", "UNNATURAL_GEOMETRY"):
            primary_access = {
                "authorized": False,
                "reason": "SPOOF_DETECTED",
                "message": "Liveness check failed: Biometric presentation attack suspected.",
            }
        elif primary["is_authorized"]:
            primary_access = {
                "authorized": True,
                "reason": "AUTHORIZED",
                "name": primary["name"],
                "participant_id": primary["participant_id"],
                "similarity": primary["similarity"],
                "message": f"Access granted for enrolled subject {primary['name']}.",
            }
        else:
            primary_access = {
                "authorized": False,
                "reason": "UNKNOWN_SUBJECT",
                "name": primary["name"],
                "similarity": primary["similarity"],
                "message": "Unregistered subject detected. Guest monitoring mode available without restrictions.",
            }

        return {
            "success": True,
            "status": "OK",
            "count": len(detections),
            "primary_access": primary_access,
            "detections": detections,
            "annotated_frame": annotated_frame,
        }

    def enroll_participant(
        self, name: str, frame: np.ndarray, notes: str = ""
    ) -> Dict[str, Any]:
        """
        Enrolls a new subject with strict quality filters (single face, resolution, pose, sharpness, illumination).
        """
        if not self.is_ready or self.detector is None or self.recognizer is None:
            return {"success": False, "message": "Face recognition pipeline not ready"}

        if frame is None or frame.size == 0:
            return {"success": False, "message": "Invalid or empty enrollment image"}

        h, w = frame.shape[:2]
        bboxes, landmarks = self.detector.detect(image=frame)

        if len(bboxes) == 0:
            return {"success": False, "message": "No face detected in enrollment image."}

        if len(bboxes) > 1:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            max_area = max(areas)
            prominent = [a for a in areas if a > 0.30 * max_area]
            if len(prominent) > 1:
                return {
                    "success": False,
                    "message": f"Multiple faces detected ({len(bboxes)} found).",
                }
            primary_idx = int(np.argmax(areas))
            bbox, landmark = bboxes[primary_idx], landmarks[primary_idx]
        else:
            bbox, landmark = bboxes[0], landmarks[0]

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_w, box_h = x2 - x1, y2 - y1

        if box_w < 60 or box_h < 60:
            return {
                "success": False,
                "message": f"Face resolution too low ({box_w}x{box_h} px). Minimum required: 60x60 px.",
            }

        pose = self.liveness_detector.estimate_head_pose(landmark)
        if abs(pose["yaw"]) > 25.0 or abs(pose["pitch"]) > 25.0 or abs(pose["roll"]) > 25.0:
            return {
                "success": False,
                "message": f"Extreme head angle detected (Yaw: {pose['yaw']}°, Pitch: {pose['pitch']}°, Roll: {pose['roll']}°).",
            }

        face_roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        luma = float(np.mean(gray_roi))
        blur_score = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

        if luma < 30.0:
            return {"success": False, "message": f"Lighting too dim ({round(luma, 1)}/255)."}

        if blur_score < 30.0:
            return {"success": False, "message": f"Image is blurry ({round(blur_score, 1)})."}

        enroll_frame = frame
        if luma < 55.0:
            enroll_frame = enhance_low_light(frame)

        aligned_face = norm_crop(img=enroll_frame, landmark=landmark)
        embedding = self.extract_embedding(aligned_face)

        participant_id = self.db.add_participant(name=name, embedding=embedding, notes=notes)
        self.reload_enrollment_dataset()

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
