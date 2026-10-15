"""
FastAPI & WebSocket Server for Multimodal Health & Access System
Integrates biomedical signal processing, hardware serial telemetry,
face detection/recognition/liveness, machine learning BP estimation,
and SQLite session persistence.
"""

import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field

from biomedical.combined import CombinedBiomedicalAnalyzer
from biomedical.ecg import ECGProcessor
from biomedical.ppg import PPGProcessor
from database.db import DatabaseManager
from face_pipeline.face_service import FaceService
from hardware.serial_reader import SerialManager
from ml.bp_model import BloodPressureFeatureExtractor, BloodPressureModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server")

# ============================================================================
# Helper Utilities & Sanitization
# ============================================================================

def sanitize_json(obj: Any) -> Any:
    """Recursively converts NumPy types, NaNs, and Infs into standard JSON-compliant primitives."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, float)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return [sanitize_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj


def decode_base64_image(b64_str: str) -> Optional[np.ndarray]:
    """Decodes a base64 string or Data URL into an OpenCV BGR numpy image."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


def encode_base64_image(frame: np.ndarray, quality: int = 80) -> str:
    """Encodes an OpenCV BGR numpy frame into a JPEG Data URL string."""
    try:
        success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return ""
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to encode frame to base64: {e}")
        return ""


# ============================================================================
# Application State Container
# ============================================================================

class AppState:
    def __init__(self, db_path: str = "health_monitor.db"):
        self.db = DatabaseManager(db_path)
        self.serial_mgr = SerialManager(port=None, baudrate=115200, mock_mode=True)
        self.face_srv = FaceService(self.db)
        self.ecg_proc = ECGProcessor(sampling_rate=100.0)
        self.ppg_proc = PPGProcessor(sampling_rate=100.0)
        self.combined_proc = CombinedBiomedicalAnalyzer(sampling_rate=100.0)

        # Auto-load Random Forest BP model from models/rf_bp_model.joblib
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rf_bp_model.joblib")
        if not os.path.exists(model_path):
            model_path = "models/rf_bp_model.joblib"
        self.bp_model = BloodPressureModel(model_file=model_path)
        self.bp_extractor = BloodPressureFeatureExtractor(sampling_rate=100.0)

        # Seed default authorized participants (Ashutosh & Dhrunil) into SQLite and FaceService
        try:
            self.db.seed_default_participants()
            self.face_srv.reload_known_faces()
        except Exception as e:
            logger.warning(f"Could not seed default participants in AppState init: {e}")

        # Authenticated user profile tracking & access control
        self.current_user: Optional[Dict[str, Any]] = None

        # Session tracking
        self.active_session_id: Optional[str] = None
        self.active_participant_id: Optional[int] = None
        self.is_guest_session: bool = False
        self.session_start_time: Optional[float] = None

        # Background broadcast & persistence state
        self.telemetry_task: Optional[asyncio.Task] = None
        self.last_sent_seq: int = -1
        self.latest_metrics: Dict[str, Any] = {}
        self.metrics_persist_counter: int = 0


state = AppState()


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages connected WebSocket telemetry clients with non-blocking broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcasts a JSON message to all connected clients, dropping dead connections."""
        async with self._lock:
            if not self.active_connections:
                return
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            for d in disconnected:
                if d in self.active_connections:
                    self.active_connections.remove(d)


manager = ConnectionManager()


# ============================================================================
# Background Telemetry & Biomedical Engine Loop
# ============================================================================

# Authorized enrolled personnel eligible for physiological calculations
AUTHORIZED_PARTICIPANTS = {"ashutosh", "dhrunil"}


def is_authorized_active() -> bool:
    """
    Checks if an authorized enrolled participant (Ashutosh or Dhrunil)
    is currently authenticated / active in the system.
    """
    if state.current_user and state.current_user.get("name"):
        if state.current_user["name"].strip().lower() in AUTHORIZED_PARTICIPANTS:
            return True

    if state.active_participant_id is not None:
        p = state.db.get_participant_by_id(state.active_participant_id)
        if not p:
            for item in state.db.get_all_participants():
                if item.get("participant_id") == state.active_participant_id:
                    p = item
                    break
        if p and p.get("name", "").strip().lower() in AUTHORIZED_PARTICIPANTS:
            return True

    return False


def compute_biomedical_metrics(samples: List[tuple], override_authorized: Optional[bool] = None) -> Dict[str, Any]:
    """
    Computes ECG, PPG, combined, and BP metrics on a sliding window.
    Designed to run efficiently in a worker thread.

    Gating: If no authorized enrolled participant (Ashutosh or Dhrunil) is active/logged in,
    marks metrics as RESTRICTED/UNREGISTERED so vitals are ONLY calculated for registered users.
    """
    authorized = override_authorized if override_authorized is not None else is_authorized_active()
    leads_off = int(samples[-1][5]) if (samples and len(samples) > 0 and len(samples[0]) > 5) else 0

    if not authorized:
        return {
            "ecg_hr": None,
            "ppg_hr": None,
            "spo2": None,
            "rr_interval_ms": None,
            "rmssd_ms": None,
            "sdnn_ms": None,
            "pat_ms": None,
            "resp_rate_bpm": None,
            "signal_quality": "RESTRICTED/UNREGISTERED",
            "bp_systolic": None,
            "bp_diastolic": None,
            "bp_status": "RESTRICTED/UNREGISTERED",
            "access_status": "RESTRICTED/UNREGISTERED",
            "is_lead_off": bool(leads_off == 1),
            "is_finger_off": False,
            "hr_concordant": False,
            "message": "Access Restricted: Physiological vitals are exclusively calculated for enrolled personnel (Ashutosh / Dhrunil).",
        }

    if not samples or len(samples) < 150:
        return {
            "ecg_hr": None,
            "ppg_hr": None,
            "spo2": None,
            "rr_interval_ms": None,
            "rmssd_ms": None,
            "sdnn_ms": None,
            "pat_ms": None,
            "resp_rate_bpm": None,
            "signal_quality": "INSUFFICIENT_DATA",
            "bp_systolic": None,
            "bp_diastolic": None,
            "bp_status": "ESTIMATED_RESEARCH_ONLY" if state.bp_model.is_trained else "UNCALIBRATED",
            "access_status": "AUTHORIZED",
            "is_lead_off": False,
            "is_finger_off": False,
            "hr_concordant": False,
        }

    arr = np.array(samples)
    ecg_raw = arr[:, 2]
    ppg_red = arr[:, 3]
    ppg_ir = arr[:, 4]
    leads_off = int(arr[-1, 5])

    ecg_res = state.ecg_proc.process(ecg_raw, leads_off=leads_off)
    ppg_res = state.ppg_proc.process(ppg_red, ppg_ir)

    ecg_hr = ecg_res.get("heart_rate")
    ppg_hr = ppg_res.get("heart_rate")

    hr_comp = state.combined_proc.compare_heart_rates(ecg_hr, ppg_hr)
    pat_res = state.combined_proc.compute_pulse_arrival_time(ecg_res.get("r_peaks"), ppg_res.get("pulse_peaks"))
    resp_res = state.combined_proc.estimate_respiratory_rate(
        ecg_res.get("filtered_ecg"), ecg_res.get("r_peaks"), ppg_res.get("filtered_ir")
    )

    bp_features = state.bp_extractor.extract_features(
        ecg_res.get("r_peaks", np.array([])),
        ecg_res.get("filtered_ecg", np.array([])),
        ppg_res.get("pulse_peaks", np.array([])),
        ppg_res.get("filtered_ir", np.array([])),
        ppg_ir
    )
    bp_res = state.bp_model.predict(bp_features) if bp_features else {
        "sbp_mmHg": None,
        "dbp_mmHg": None,
        "status": "ESTIMATED_RESEARCH_ONLY" if state.bp_model.is_trained else "UNCALIBRATED",
    }

    is_lead_off = bool(leads_off == 1 or ecg_raw[-1] >= 1000 or ecg_res["signal_quality"]["quality"] == "DISCONNECTED")
    is_finger_off = bool(not ppg_res["contact_status"]["contact"])

    rr_list = ecg_res.get("rr_intervals_ms", [])
    latest_rr = float(rr_list[-1]) if len(rr_list) > 0 else None

    return {
        "ecg_hr": ecg_hr,
        "ppg_hr": ppg_hr,
        "spo2": ppg_res.get("spo2"),
        "rr_interval_ms": latest_rr,
        "rmssd_ms": ecg_res.get("hrv", {}).get("rmssd_ms"),
        "sdnn_ms": ecg_res.get("hrv", {}).get("sdnn_ms"),
        "pat_ms": pat_res.get("pat_median_ms"),
        "resp_rate_bpm": resp_res.get("respiratory_rate_bpm"),
        "signal_quality": ecg_res.get("signal_quality", {}).get("quality", "N/A"),
        "bp_systolic": bp_res.get("sbp_mmHg"),
        "bp_diastolic": bp_res.get("dbp_mmHg"),
        "bp_status": bp_res.get("status", "UNCALIBRATED"),
        "access_status": "AUTHORIZED",
        "is_lead_off": is_lead_off,
        "is_finger_off": is_finger_off,
        "hr_concordant": bool(hr_comp.get("is_concordant", False)),
    }


async def telemetry_broadcaster_loop():
    """
    Continuous async loop streaming live telemetry and computed window metrics.
    Runs at 20 Hz (every 50ms), persisting sensor samples if a session is active.
    """
    logger.info("Telemetry broadcaster background task started.")
    last_seq = -1

    while True:
        try:
            # 1. Fetch recent samples (sliding window of 600 samples = 6s at 100Hz)
            samples = state.serial_mgr.get_latest_samples(600)
            new_samples = []

            if samples:
                # Identify newly arrived samples based on sequence number
                if last_seq == -1:
                    new_samples = samples[-10:]
                else:
                    new_samples = [s for s in samples if s[1] > last_seq or (s[1] < last_seq and (last_seq - s[1]) > 50000)]
                    if not new_samples and len(samples) > 0:
                        # Fallback if seq wrapped or uninitialized
                        new_samples = samples[-2:]

                if samples:
                    last_seq = samples[-1][1]

            # 2. Persist raw samples if session is active
            if state.active_session_id and new_samples:
                to_persist = [(state.active_session_id, s[0], s[1], s[2], s[3], s[4], s[5]) for s in new_samples]
                try:
                    await asyncio.to_thread(state.db.insert_sensor_samples, to_persist)
                except Exception as e:
                    logger.error(f"Error persisting sensor telemetry: {e}")

            # 3. Compute biomedical metrics off-thread
            if len(samples) >= 150:
                metrics = await asyncio.to_thread(compute_biomedical_metrics, samples)
            else:
                metrics = compute_biomedical_metrics([])
            state.latest_metrics = metrics

            # 4. Periodic metrics persistence (every ~1 second = 20 iterations)
            state.metrics_persist_counter += 1
            if state.active_session_id and state.metrics_persist_counter % 20 == 0:
                metric_row = {
                    "timestamp_s": time.time(),
                    "ecg_hr": metrics["ecg_hr"],
                    "ppg_hr": metrics["ppg_hr"],
                    "spo2": metrics["spo2"],
                    "rr_interval_ms": metrics["rr_interval_ms"],
                    "rmssd_ms": metrics["rmssd_ms"],
                    "sdnn_ms": metrics["sdnn_ms"],
                    "pat_ms": metrics["pat_ms"],
                    "resp_rate_bpm": metrics["resp_rate_bpm"],
                    "signal_quality": metrics["signal_quality"],
                    "bp_systolic": metrics["bp_systolic"],
                    "bp_diastolic": metrics["bp_diastolic"],
                }
                try:
                    await asyncio.to_thread(state.db.insert_metric, state.active_session_id, metric_row)
                except Exception as e:
                    logger.error(f"Error persisting metrics: {e}")

            # 5. Broadcast to connected WebSocket clients
            if manager.active_connections and new_samples:
                payload = {
                    "type": "telemetry",
                    "timestamp": time.time(),
                    "samples": [
                        {
                            "timestamp_ms": s[0],
                            "seq": s[1],
                            "ecg_raw": s[2],
                            "ppg_red": s[3],
                            "ppg_ir": s[4],
                            "leads_off": s[5],
                        }
                        for s in new_samples
                    ],
                    "metrics": sanitize_json(metrics),
                    "session": {
                        "active_session_id": state.active_session_id,
                        "participant_id": state.active_participant_id,
                        "is_guest": state.is_guest_session,
                        "is_authorized": is_authorized_active(),
                        "current_user": state.current_user,
                    },
                    "serial_status": {
                        "status": state.serial_mgr.connection_status,
                        "mock_mode": state.serial_mgr.mock_mode,
                        "port": state.serial_mgr.port,
                        "dropped_packets": state.serial_mgr.dropped_packets,
                    },
                }
                await manager.broadcast(payload)

            await asyncio.sleep(0.05)  # 50ms = 20Hz update rate
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in telemetry broadcast loop: {e}", exc_info=True)
            await asyncio.sleep(0.1)


# ============================================================================
# FastAPI Lifespan (Startup & Graceful Shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup service initialization and graceful shutdown."""
    logger.info("Initializing Multimodal Health & Access System Server...")

    # 1. Ensure Ashutosh and Dhrunil exist in database and are actively loaded into FaceService
    try:
        state.db.seed_default_participants()
        state.face_srv.reload_known_faces()
        logger.info(f"FaceService loaded enrolled participants: {state.face_srv.known_names}")
    except Exception as e:
        logger.error(f"Error initializing FaceService enrolled participants: {e}")

    # 2. Ensure BloodPressureModel automatically loads models/rf_bp_model.joblib on startup
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rf_bp_model.joblib")
    if not state.bp_model.is_trained and os.path.exists(model_path):
        state.bp_model.load_model(model_path)
    logger.info(f"BloodPressureModel loaded on startup. Ready: {state.bp_model.is_trained}")

    # 3. Start serial acquisition (defaults to mock_mode=True for standalone operation)
    state.serial_mgr.start()

    # 4. Start non-blocking telemetry broadcast task
    state.telemetry_task = asyncio.create_task(telemetry_broadcaster_loop())

    yield

    # Graceful Shutdown
    logger.info("Shutting down server...")
    if state.telemetry_task:
        state.telemetry_task.cancel()
        try:
            await state.telemetry_task
        except asyncio.CancelledError:
            pass

    # Complete any active session cleanly
    if state.active_session_id:
        try:
            state.db.end_session(state.active_session_id, {"reason": "graceful_server_shutdown"})
            logger.info(f"Closed active session {state.active_session_id} on shutdown.")
        except Exception as e:
            logger.error(f"Error finalizing session on shutdown: {e}")

    state.serial_mgr.stop()
    logger.info("Server shutdown complete.")


# Initialize FastAPI App
app = FastAPI(
    title="ESA Multimodal Biometric & Telemetry Cockpit API",
    version="2.0.0",
    lifespan=lifespan,
    description="High-frequency medical telemetry, face access verification, and session persistence engine."
)

# CORS Middleware (Universal Access for Web Clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Request Models
# ============================================================================

class FaceProcessRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded camera frame or Data URL")
    recognition_thresh: float = Field(0.35, ge=0.0, le=1.0)
    return_annotated: bool = Field(True, description="Whether to return annotated base64 frame")


class FaceEnrollRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Participant name or unique subject identifier")
    image: str = Field(..., description="Base64 encoded face photo for registration")
    notes: Optional[str] = Field("", description="Optional clinical notes or demographic metadata")


class FaceLoginRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded camera frame or Data URL")
    recognition_thresh: float = Field(0.35, ge=0.0, le=1.0)


class DirectLoginRequest(BaseModel):
    name: Optional[str] = None
    participant_id: Optional[int] = None


class ParticipantCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    notes: Optional[str] = ""
    image: Optional[str] = None


class ParticipantDeleteRequest(BaseModel):
    participant_id: int


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = None
    participant_id: Optional[int] = None
    is_guest: bool = False


class SessionStopRequest(BaseModel):
    session_id: Optional[str] = None


class SessionDeleteRequest(BaseModel):
    session_id: str


class SerialSettingsRequest(BaseModel):
    port: Optional[str] = None
    baudrate: Optional[int] = None
    mock_mode: Optional[bool] = None
    sim_heart_rate: Optional[float] = None
    sim_leads_off: Optional[int] = None
    sim_finger_contact: Optional[bool] = None
    connect: Optional[bool] = None


class SimulatorSettingsRequest(BaseModel):
    heart_rate: Optional[float] = None
    leads_off: Optional[int] = None
    finger_contact: Optional[bool] = None


# ============================================================================
# 1. WEBSOCKET TELEMETRY ENDPOINT
# ============================================================================

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint streaming live 100Hz biomedical samples,
    computed window metrics, lead/finger contact statuses, and receiving client commands.
    """
    await manager.connect(websocket)

    # Immediately push initial state packet
    initial_packet = {
        "type": "init",
        "timestamp": time.time(),
        "session": {
            "active_session_id": state.active_session_id,
            "participant_id": state.active_participant_id,
            "is_guest": state.is_guest_session,
            "is_authorized": is_authorized_active(),
            "current_user": state.current_user,
        },
        "serial_status": {
            "status": state.serial_mgr.connection_status,
            "mock_mode": state.serial_mgr.mock_mode,
            "port": state.serial_mgr.port,
        },
        "metrics": sanitize_json(state.latest_metrics),
    }
    try:
        await websocket.send_json(initial_packet)
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("command")

                if cmd == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})

                elif cmd == "login":
                    name = msg.get("name", "")
                    if name and name.strip().lower() == "guest":
                        state.current_user = {
                            "participant_id": None,
                            "name": "Guest Subject",
                            "role": "Guest Reviewer",
                            "avatar": "GU",
                        }
                        state.active_participant_id = None
                        state.is_guest_session = True
                        await websocket.send_json({
                            "type": "auth_status",
                            "authenticated": False,
                            "is_guest": True,
                            "user": state.current_user,
                        })
                    elif name and name.strip().lower() in AUTHORIZED_PARTICIPANTS:
                        canon = "Ashutosh" if name.strip().lower() == "ashutosh" else "Dhrunil"
                        p = state.db.get_participant_by_name(canon)
                        part_id = p["participant_id"] if p else (16 if canon == "Ashutosh" else 17)
                        state.current_user = {
                            "participant_id": part_id,
                            "name": canon,
                            "role": "Lead Biomedical Research Engineer" if canon == "Ashutosh" else "Lead Systems & Telemetry Architect",
                            "avatar": "AS" if canon == "Ashutosh" else "DH",
                        }
                        state.active_participant_id = part_id
                        state.is_guest_session = False
                        await websocket.send_json({
                            "type": "auth_status",
                            "authenticated": True,
                            "user": state.current_user,
                        })
                    else:
                        await websocket.send_json({
                            "type": "auth_status",
                            "authenticated": False,
                            "reason": "UNREGISTERED_USER",
                        })

                elif cmd == "logout":
                    state.current_user = None
                    state.active_participant_id = None
                    if state.active_session_id:
                        state.db.end_session(state.active_session_id, {"reason": "user_logout"})
                        state.active_session_id = None
                    state.latest_metrics = compute_biomedical_metrics([], override_authorized=False)
                    await websocket.send_json({
                        "type": "auth_status",
                        "authenticated": False,
                        "user": None,
                    })

                elif cmd == "start_session":
                    sess_id = msg.get("session_id") or f"SES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    part_id = msg.get("participant_id")
                    is_guest = bool(msg.get("is_guest", False))
                    state.db.create_session(sess_id, part_id, is_guest)
                    state.active_session_id = sess_id
                    state.active_participant_id = part_id
                    state.is_guest_session = is_guest
                    state.session_start_time = time.time()
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": sess_id,
                        "participant_id": part_id,
                        "is_guest": is_guest
                    })

                elif cmd == "stop_session":
                    if state.active_session_id:
                        sess_id = state.active_session_id
                        metrics_list = state.db.get_session_metrics(sess_id)
                        summary = {
                            "metrics_count": len(metrics_list),
                            "duration_seconds": (time.time() - state.session_start_time) if state.session_start_time else 0.0,
                        }
                        state.db.end_session(sess_id, summary)
                        state.active_session_id = None
                        state.active_participant_id = None
                        state.is_guest_session = False
                        await websocket.send_json({"type": "session_stopped", "session_id": sess_id, "summary": summary})

                elif cmd == "set_mock_mode":
                    enabled = bool(msg.get("enabled", True))
                    state.serial_mgr.set_mock_mode(enabled)
                    await websocket.send_json({"type": "status", "mock_mode": state.serial_mgr.mock_mode})

                elif cmd == "set_sim_controls":
                    hr = float(msg.get("heart_rate", state.serial_mgr._sim_heart_rate))
                    lo = int(msg.get("leads_off", state.serial_mgr._sim_leads_off))
                    fc = bool(msg.get("finger_contact", state.serial_mgr._sim_finger_contact))
                    state.serial_mgr.set_simulator_controls(heart_rate=hr, leads_off=lo, finger_contact=fc)
                    await websocket.send_json({"type": "sim_controls_updated", "heart_rate": hr, "leads_off": lo, "finger_contact": fc})

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket client error: {e}")
        await manager.disconnect(websocket)


# ============================================================================
# 2. AUTHENTICATION & ACCESS CONTROL API
# ============================================================================

@app.get("/api/auth/current")
def get_current_authenticated_user():
    """
    Returns current authenticated profile details and physiological access status.
    """
    authorized = is_authorized_active()
    return {
        "authenticated": authorized,
        "user": state.current_user if authorized else None,
        "authorized_participants": ["Ashutosh", "Dhrunil"],
        "access_status": "AUTHORIZED" if authorized else "RESTRICTED/UNREGISTERED"
    }


@app.post("/api/auth/login-by-face")
async def login_by_face(req: FaceLoginRequest):
    """
    Authenticates an enrolled subject (Ashutosh or Dhrunil) via live camera frame.
    Enforces anti-spoofing liveness check and biometric ArcFace recognition.
    """
    frame = decode_base64_image(req.image)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image payload. Expected valid base64 image data.")

    result = await asyncio.to_thread(
        state.face_srv.process_frame,
        frame=frame,
        recognition_thresh=req.recognition_thresh
    )

    detections = result.get("detections", [])
    if not detections:
        return {
            "success": False,
            "authenticated": False,
            "reason": "NO_FACE",
            "message": "No face detected in camera frame. Position your face directly towards the camera.",
            "detections": []
        }

    primary = detections[0]

    name = primary.get("name", "").strip()
    is_authorized_name = name.lower() in AUTHORIZED_PARTICIPANTS

    # Anti-spoofing check: only block if explicitly confirmed attack
    liveness = primary.get("liveness", {})
    liveness_status = liveness.get("status", "ANALYZING")
    if liveness_status == "ATTACK_SUSPECTED":
        return {
            "success": False,
            "authenticated": False,
            "reason": "SPOOF_DETECTED",
            "message": "Liveness check failed. Biometric photo attack suspected.",
            "detection": sanitize_json(primary),
            "detections": sanitize_json(detections)
        }

    if not is_authorized_name:
        return {
            "success": False,
            "authenticated": False,
            "reason": "UNKNOWN_SUBJECT",
            "name": name if name else "UNKNOWN",
            "similarity": primary.get("similarity", 0.0),
            "message": "Unregistered subject detected (UNKNOWN). Vital calculations are calibrated for enrolled personnel (Ashutosh & Dhrunil).",
            "detection": sanitize_json(primary),
            "detections": sanitize_json(detections)
        }

    canon_name = "Ashutosh" if name.lower() == "ashutosh" else "Dhrunil"
    p = state.db.get_participant_by_name(canon_name)
    part_id = p["participant_id"] if p else (16 if canon_name == "Ashutosh" else 17)
    role = "Lead Biomedical Research Engineer" if canon_name == "Ashutosh" else "Lead Systems & Telemetry Architect"
    avatar = "AS" if canon_name == "Ashutosh" else "DH"

    user_profile = {
        "participant_id": part_id,
        "name": canon_name,
        "role": role,
        "avatar": avatar,
        "similarity": primary.get("similarity", 1.0),
        "notes": p.get("notes", "") if p else "",
        "enrolled_at": p.get("enrolled_at", "") if p else ""
    }

    state.current_user = user_profile
    state.active_participant_id = part_id
    state.is_guest_session = False

    return {
        "success": True,
        "authenticated": True,
        "user": user_profile,
        "message": f"Welcome back, {canon_name}!",
        "detection": sanitize_json(primary),
        "detections": sanitize_json(detections)
    }


@app.post("/api/auth/login")
def direct_login(req: DirectLoginRequest):
    """
    Direct login endpoint for enrolled personnel (Ashutosh / Dhrunil) and Guest.
    Useful for UI personnel switcher, automated testing, and development.
    """
    target_name = None
    if req.name:
        target_name = req.name.strip()
    elif req.participant_id:
        p = state.db.get_participant_by_id(req.participant_id)
        if p:
            target_name = p["name"]

    if target_name and target_name.lower() == "guest":
        user_profile = {
            "participant_id": None,
            "name": "Guest Subject",
            "role": "Guest Reviewer",
            "avatar": "GU",
            "notes": "Guest session. Waveforms visible. Vital calculations calibrated for Ashutosh & Dhrunil.",
            "enrolled_at": ""
        }
        state.current_user = user_profile
        state.active_participant_id = None
        state.is_guest_session = True
        return {
            "success": True,
            "authenticated": False,
            "is_guest": True,
            "user": user_profile,
            "message": "Logged in as Guest"
        }

    if not target_name or target_name.lower() not in AUTHORIZED_PARTICIPANTS:
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: Access restricted exclusively to enrolled personnel (Ashutosh & Dhrunil)."
        )

    canon_name = "Ashutosh" if target_name.lower() == "ashutosh" else "Dhrunil"
    p = state.db.get_participant_by_name(canon_name)
    part_id = p["participant_id"] if p else (16 if canon_name == "Ashutosh" else 17)
    role = "Lead Biomedical Research Engineer" if canon_name == "Ashutosh" else "Lead Systems & Telemetry Architect"
    avatar = "AS" if canon_name == "Ashutosh" else "DH"

    user_profile = {
        "participant_id": part_id,
        "name": canon_name,
        "role": role,
        "avatar": avatar,
        "notes": p.get("notes", "") if p else "",
        "enrolled_at": p.get("enrolled_at", "") if p else ""
    }

    state.current_user = user_profile
    state.active_participant_id = part_id
    state.is_guest_session = False

    return {
        "success": True,
        "authenticated": True,
        "user": user_profile,
        "message": f"Logged in as {canon_name}"
    }


@app.post("/api/auth/logout")
def logout_user():
    """
    Logs out the active user, locking physiological calculations back to RESTRICTED.
    """
    state.current_user = None
    state.active_participant_id = None
    if state.active_session_id:
        state.db.end_session(state.active_session_id, {"reason": "user_logout"})
        state.active_session_id = None
        state.session_start_time = None

    state.latest_metrics = compute_biomedical_metrics([], override_authorized=False)

    return {
        "success": True,
        "authenticated": False,
        "message": "User logged out successfully. Access locked."
    }


# ============================================================================
# 3. FACE ACCESS & BIOMETRICS API
# ============================================================================

@app.post("/api/face/process")
async def face_process_frame(req: FaceProcessRequest):
    """
    Analyzes a base64 camera frame for face detection, tracking, liveness verification,
    and ArcFace identity matching against enrolled participants.
    """
    frame = decode_base64_image(req.image)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image payload. Expected valid base64 image data.")

    # Execute face pipeline off-thread to avoid blocking event loop
    result = await asyncio.to_thread(
        state.face_srv.process_frame,
        frame=frame,
        recognition_thresh=req.recognition_thresh
    )

    annotated_b64 = None
    if req.return_annotated and result.get("annotated_frame") is not None:
        annotated_b64 = encode_base64_image(result["annotated_frame"])

    detections = sanitize_json(result.get("detections", []))

    return {
        "status": result.get("status", "OK"),
        "count": len(detections),
        "detections": detections,
        "annotated_image": annotated_b64,
    }


@app.post("/api/face/enroll")
async def face_enroll_participant(req: FaceEnrollRequest):
    """
    Registers a new participant from a base64 camera frame into the biometric database.
    Extracts 512-dim ArcFace embedding with pose and lighting quality checks.
    """
    clean_name = req.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Participant name cannot be empty.")

    frame = decode_base64_image(req.image)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Invalid enrollment image payload.")

    res = await asyncio.to_thread(
        state.face_srv.enroll_participant,
        name=clean_name,
        frame=frame,
        notes=req.notes or ""
    )

    participant_record = None
    if res.get("success"):
        p = state.db.get_participant_by_name(clean_name)
        if p:
            participant_record = {
                "participant_id": p["participant_id"],
                "name": p["name"],
                "enrolled_at": p["enrolled_at"],
                "notes": p.get("notes", ""),
                "has_embedding": bool(p.get("embedding") is not None),
            }

    return {
        "success": res.get("success", False),
        "message": res.get("message", ""),
        "participant": participant_record,
    }


# ============================================================================
# 3. PARTICIPANTS CRUD API
# ============================================================================

@app.get("/api/participants")
def get_participants():
    """Lists all enrolled participants with registration metadata."""
    participants = state.db.get_all_participants()
    result = []
    for p in participants:
        result.append({
            "participant_id": p["participant_id"],
            "name": p["name"],
            "enrolled_at": p["enrolled_at"],
            "notes": p.get("notes", ""),
            "has_embedding": bool(p.get("embedding") is not None),
        })
    return result


@app.post("/api/participants")
def create_participant(req: ParticipantCreateRequest):
    """Creates or enrolls a participant record (with optional facial embedding)."""
    clean_name = req.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Participant name cannot be empty.")

    if req.image:
        frame = decode_base64_image(req.image)
        if frame is not None and frame.size > 0:
            enroll_res = state.face_srv.enroll_participant(clean_name, frame, notes=req.notes or "")
            if not enroll_res["success"]:
                return JSONResponse(status_code=400, content={"success": False, "message": enroll_res["message"]})
            p = state.db.get_participant_by_name(clean_name)
            return {"success": True, "participant_id": p["participant_id"] if p else None, "name": clean_name}

    # Add text-only participant if no image provided
    part_id = state.db.add_participant(clean_name, notes=req.notes or "")
    return {"success": True, "participant_id": part_id, "name": clean_name}


@app.delete("/api/participants/{participant_id}")
def delete_participant_by_path(participant_id: int):
    """Deletes participant and associated biometric record by ID."""
    deleted = state.db.delete_participant(participant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Participant {participant_id} not found.")
    state.face_srv.reload_known_faces()
    return {"success": True, "deleted_id": participant_id}


@app.delete("/api/participants")
def delete_participant_by_query(
    id: Optional[int] = Query(None),
    participant_id: Optional[int] = Query(None),
    payload: Optional[ParticipantDeleteRequest] = None
):
    """Flexible participant deletion accepting query parameter or JSON body."""
    target_id = id or participant_id or (payload.participant_id if payload else None)
    if target_id is None:
        raise HTTPException(status_code=400, detail="Missing participant_id parameter.")
    deleted = state.db.delete_participant(target_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Participant {target_id} not found.")
    state.face_srv.reload_known_faces()
    return {"success": True, "deleted_id": target_id}


# ============================================================================
# 4. SESSIONS MANAGEMENT API
# ============================================================================

@app.get("/api/sessions")
def get_sessions():
    """Retrieves all telemetry recording sessions, including current active session indicator."""
    sessions = state.db.get_all_sessions()
    return {
        "active_session_id": state.active_session_id,
        "is_guest": state.is_guest_session,
        "sessions": sessions,
    }


@app.get("/api/sessions/active")
def get_active_session():
    """Returns details of the currently running session if any."""
    if not state.active_session_id:
        return {"active": False, "session": None}
    session = state.db.get_session(state.active_session_id)
    return {
        "active": True,
        "session": session,
        "duration_seconds": (time.time() - state.session_start_time) if state.session_start_time else 0.0,
    }


@app.get("/api/sessions/{session_id}")
def get_session_details(session_id: str):
    """Retrieves single session details and computed metrics history."""
    session = state.db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    metrics = state.db.get_session_metrics(session_id)
    return {"session": session, "metrics": metrics}


@app.post("/api/sessions")
def start_session(req: SessionCreateRequest):
    """Starts a new recording session and marks it as active."""
    # If a session is already running, finalize it first
    if state.active_session_id:
        state.db.end_session(state.active_session_id, {"reason": "preempted_by_new_session"})

    sess_id = req.session_id or f"SES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    created_id = state.db.create_session(sess_id, req.participant_id, req.is_guest)

    state.active_session_id = created_id
    state.active_participant_id = req.participant_id
    state.is_guest_session = req.is_guest
    state.session_start_time = time.time()

    return {
        "success": True,
        "session_id": created_id,
        "status": "ACTIVE",
        "participant_id": req.participant_id,
        "is_guest": req.is_guest,
    }


@app.post("/api/sessions/stop")
def stop_active_session(req: Optional[SessionStopRequest] = None):
    """Stops the active session and computes summary diagnostics."""
    target_id = (req.session_id if req and req.session_id else None) or state.active_session_id
    if not target_id:
        raise HTTPException(status_code=400, detail="No session currently active to stop.")

    metrics = state.db.get_session_metrics(target_id)
    duration = (time.time() - state.session_start_time) if state.session_start_time else 0.0

    # Calculate summary aggregations
    ecg_hrs = [m["ecg_hr"] for m in metrics if m.get("ecg_hr") is not None]
    spo2s = [m["spo2"] for m in metrics if m.get("spo2") is not None]
    sbps = [m["bp_systolic"] for m in metrics if m.get("bp_systolic") is not None]
    dbps = [m["bp_diastolic"] for m in metrics if m.get("bp_diastolic") is not None]

    summary = {
        "metrics_count": len(metrics),
        "duration_seconds": round(duration, 1),
        "mean_ecg_hr": round(float(np.mean(ecg_hrs)), 1) if ecg_hrs else None,
        "mean_spo2": round(float(np.mean(spo2s)), 1) if spo2s else None,
        "mean_sbp": round(float(np.mean(sbps)), 1) if sbps else None,
        "mean_dbp": round(float(np.mean(dbps)), 1) if dbps else None,
    }

    state.db.end_session(target_id, summary)

    if state.active_session_id == target_id:
        state.active_session_id = None
        state.active_participant_id = None
        state.is_guest_session = False
        state.session_start_time = None

    return {"success": True, "session_id": target_id, "summary": summary}


@app.post("/api/sessions/end")
def end_active_session(req: Optional[SessionStopRequest] = None):
    """Alias for /api/sessions/stop."""
    return stop_active_session(req)


@app.delete("/api/sessions/{session_id}")
def delete_session_by_path(session_id: str):
    """Deletes recording session and cascades associated timeseries/metrics."""
    if state.active_session_id == session_id:
        state.active_session_id = None
        state.active_participant_id = None
        state.is_guest_session = False
        state.session_start_time = None

    deleted = state.db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return {"success": True, "deleted_session_id": session_id}


@app.delete("/api/sessions")
def delete_session_by_query(
    session_id: Optional[str] = Query(None),
    payload: Optional[SessionDeleteRequest] = None
):
    """Flexible session deletion endpoint accepting query parameter or JSON body."""
    target_id = session_id or (payload.session_id if payload else None)
    if not target_id:
        raise HTTPException(status_code=400, detail="Missing session_id parameter.")
    return delete_session_by_path(target_id)


# ============================================================================
# 5. DATA EXPORT & REPORTS API
# ============================================================================

@app.get("/api/reports/export/timeseries/{session_id}")
def export_timeseries(session_id: str):
    """Exports raw 100Hz sensor telemetry (ECG, Red PPG, IR PPG) for a session as CSV."""
    session = state.db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    csv_data = state.db.export_session_timeseries_csv(session_id)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="timeseries_{session_id}.csv"'}
    )


@app.get("/api/reports/export/metrics/{session_id}")
def export_metrics(session_id: str):
    """Exports computed physiological metrics (HR, SpO2, HRV, PAT, Respiration, BP) as CSV."""
    session = state.db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    csv_data = state.db.export_session_metrics_csv(session_id)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="metrics_{session_id}.csv"'}
    )


# ============================================================================
# 6. HARDWARE & SERIAL SETTINGS API
# ============================================================================

@app.get("/api/settings/serial")
def get_serial_settings():
    """Retrieves current hardware serial and simulation configuration."""
    return {
        "available_ports": SerialManager.list_available_ports(),
        "port": state.serial_mgr.port,
        "baudrate": state.serial_mgr.baudrate,
        "mock_mode": state.serial_mgr.mock_mode,
        "connection_status": state.serial_mgr.connection_status,
        "last_error": state.serial_mgr.last_error,
        "dropped_packets": state.serial_mgr.dropped_packets,
        "sim_heart_rate": state.serial_mgr._sim_heart_rate,
        "sim_leads_off": state.serial_mgr._sim_leads_off,
        "sim_finger_contact": state.serial_mgr._sim_finger_contact,
    }


@app.post("/api/settings/serial")
def update_serial_settings(req: SerialSettingsRequest):
    """Updates serial connection parameters, switches mock mode, or adjusts simulation values."""
    if req.mock_mode is not None and req.mock_mode != state.serial_mgr.mock_mode:
        state.serial_mgr.set_mock_mode(req.mock_mode)

    if req.port is not None:
        state.serial_mgr.port = None if req.port in ("AUTO", "", None) else req.port

    if req.baudrate is not None:
        state.serial_mgr.baudrate = req.baudrate

    if req.sim_heart_rate is not None or req.sim_leads_off is not None or req.sim_finger_contact is not None:
        hr = req.sim_heart_rate if req.sim_heart_rate is not None else state.serial_mgr._sim_heart_rate
        lo = req.sim_leads_off if req.sim_leads_off is not None else state.serial_mgr._sim_leads_off
        fc = req.sim_finger_contact if req.sim_finger_contact is not None else state.serial_mgr._sim_finger_contact
        state.serial_mgr.set_simulator_controls(heart_rate=hr, leads_off=lo, finger_contact=fc)

    if req.connect is True and not state.serial_mgr.mock_mode:
        state.serial_mgr.stop()
        state.serial_mgr.start()

    return get_serial_settings()


@app.get("/api/status")
def get_system_status():
    """Retrieves high-level hardware, session, and participant status for UI status bars."""
    return {
        "status": "ONLINE",
        "serial": {
            "port": state.serial_mgr.port,
            "mock_mode": state.serial_mgr.mock_mode,
            "connection_status": state.serial_mgr.connection_status,
            "last_error": state.serial_mgr.last_error,
        },
        "active_session_id": state.active_session_id,
        "active_participant_id": state.active_participant_id,
        "is_guest_session": state.is_guest_session,
        "is_authorized": is_authorized_active(),
        "current_user": state.current_user,
        "active_connections": len(manager.active_connections),
    }


@app.post("/api/settings/simulator")
def update_simulator_settings(req: SimulatorSettingsRequest):
    """Direct shortcut endpoint for updating hardware simulation parameters."""
    hr = req.heart_rate if req.heart_rate is not None else state.serial_mgr._sim_heart_rate
    lo = req.leads_off if req.leads_off is not None else state.serial_mgr._sim_leads_off
    fc = req.finger_contact if req.finger_contact is not None else state.serial_mgr._sim_finger_contact
    state.serial_mgr.set_simulator_controls(heart_rate=hr, leads_off=lo, finger_contact=fc)
    return {
        "success": True,
        "sim_heart_rate": hr,
        "sim_leads_off": lo,
        "sim_finger_contact": fc,
    }


# ============================================================================
# 7. STATIC FILES MOUNT (Cockpit Web Dashboard)

# ============================================================================

# Ensure web directory exists before mounting
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB_DIR, exist_ok=True)

# Mount web directory at root path '/'
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


# ============================================================================
# Main Execution Entrypoint
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=True)
