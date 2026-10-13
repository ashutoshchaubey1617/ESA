"""
Tests for FastAPI Web Server & WebSocket Telemetry (server.py)
Validates:
- Root index.html serving and static asset delivery
- GET /api/participants, POST /api/participants, DELETE /api/participants
- POST /api/sessions and GET /api/sessions (lifecycle, active, details, stop, delete)
- Serial settings endpoints (GET/POST /api/settings/serial, simulator, status)
- Telemetry WebSocket (/ws/telemetry) connection, streaming, and bidirectional commands
- Face processing and enrollment endpoints
- Timeseries and metrics CSV export endpoints
"""

import base64
import time
from typing import Generator
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from server import app, state


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Module-scoped FastAPI TestClient running startup & graceful shutdown lifespans."""
    with TestClient(app) as test_client:
        yield test_client


def generate_synthetic_face_base64() -> str:
    """Generates a synthetic base64 image containing facial geometry."""
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 128
    cv2.ellipse(img, (150, 150), (65, 85), 0, 0, 360, (210, 210, 210), -1)
    cv2.circle(img, (130, 130), 8, (40, 40, 40), -1)
    cv2.circle(img, (170, 130), 8, (40, 40, 40), -1)
    cv2.ellipse(img, (150, 185), (20, 8), 0, 0, 360, (40, 40, 40), -1)

    _, buffer = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ============================================================================
# 1. Root index.html Serving & Static Assets
# ============================================================================

def test_root_index_html_serving(client: TestClient):
    """Verifies that GET / serves the web cockpit HTML5 dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    content = response.text
    assert "<!DOCTYPE html>" in content or "<html" in content.lower()
    assert "ESA" in content or "Cockpit" in content or "Multimodal" in content


def test_index_html_direct_route(client: TestClient):
    """Verifies that GET /index.html also returns the HTML dashboard."""
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


# ============================================================================
# 2. Participants API Endpoints (GET /api/participants, POST, DELETE)
# ============================================================================

def test_get_participants_empty_or_list(client: TestClient):
    """Verifies GET /api/participants returns a list of registered participants."""
    response = client.get("/api/participants")
    assert response.status_code == 200
    participants = response.json()
    assert isinstance(participants, list)


def test_participant_crud_lifecycle(client: TestClient):
    """Verifies creating, fetching, and deleting a participant."""
    test_name = f"Test Participant {int(time.time())}"
    test_notes = "Astronaut Medical Flight Candidate"

    # 1. Create participant
    create_resp = client.post(
        "/api/participants",
        json={"name": test_name, "notes": test_notes}
    )
    assert create_resp.status_code == 200
    create_data = create_resp.json()
    assert create_data.get("success") is True
    part_id = create_data.get("participant_id")
    assert part_id is not None

    # 2. Verify participant appears in GET /api/participants
    list_resp = client.get("/api/participants")
    assert list_resp.status_code == 200
    all_parts = list_resp.json()
    found = next((p for p in all_parts if p["participant_id"] == part_id), None)
    assert found is not None
    assert found["name"] == test_name
    assert found["notes"] == test_notes

    # 3. Delete participant
    del_resp = client.delete(f"/api/participants/{part_id}")
    assert del_resp.status_code == 200
    assert del_resp.json().get("success") is True

    # 4. Verify participant is removed
    list_after = client.get("/api/participants").json()
    assert not any(p["participant_id"] == part_id for p in list_after)


# ============================================================================
# 3. Sessions API Endpoints (POST /api/sessions, GET /api/sessions)
# ============================================================================

def test_create_and_get_sessions(client: TestClient):
    """Verifies POST /api/sessions and GET /api/sessions lifecycle."""
    session_id = f"TEST_SES_UNIT_{int(time.time())}"

    # 1. Start a session
    post_resp = client.post(
        "/api/sessions",
        json={"session_id": session_id, "participant_id": None, "is_guest": True}
    )
    assert post_resp.status_code == 200
    data = post_resp.json()
    assert data.get("success") is True
    assert data.get("session_id") == session_id
    assert data.get("status") == "ACTIVE"
    assert data.get("is_guest") is True

    # 2. Verify GET /api/sessions returns active session state
    get_resp = client.get("/api/sessions")
    assert get_resp.status_code == 200
    sessions_data = get_resp.json()
    assert sessions_data.get("active_session_id") == session_id
    assert sessions_data.get("is_guest") is True
    assert isinstance(sessions_data.get("sessions"), list)
    assert any(s["session_id"] == session_id for s in sessions_data["sessions"])

    # 3. Verify GET /api/sessions/active
    active_resp = client.get("/api/sessions/active")
    assert active_resp.status_code == 200
    active_data = active_resp.json()
    assert active_data.get("active") is True
    assert active_data.get("session", {}).get("session_id") == session_id

    # 4. Verify GET /api/sessions/{session_id}
    detail_resp = client.get(f"/api/sessions/{session_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data.get("session", {}).get("session_id") == session_id
    assert "metrics" in detail_data

    # 5. Stop session via POST /api/sessions/stop
    stop_resp = client.post("/api/sessions/stop", json={"session_id": session_id})
    assert stop_resp.status_code == 200
    assert stop_resp.json().get("success") is True

    # 6. Verify active session is now cleared
    active_cleared = client.get("/api/sessions/active").json()
    assert active_cleared.get("active") is False

    # 7. Cleanup session
    del_resp = client.delete(f"/api/sessions/{session_id}")
    assert del_resp.status_code == 200


def test_session_stop_alias_endpoint(client: TestClient):
    """Verifies that POST /api/sessions/end works as an alias for stop."""
    sess_id = f"TEST_SES_ALIAS_{int(time.time())}"
    client.post("/api/sessions", json={"session_id": sess_id, "is_guest": True})

    stop_resp = client.post("/api/sessions/end")
    assert stop_resp.status_code == 200
    assert stop_resp.json().get("success") is True

    # Cleanup
    client.delete(f"/api/sessions/{sess_id}")


# ============================================================================
# 4. Hardware & Serial Settings Endpoints
# ============================================================================

def test_get_serial_settings(client: TestClient):
    """Verifies GET /api/settings/serial returns serial configuration."""
    response = client.get("/api/settings/serial")
    assert response.status_code == 200
    settings = response.json()
    assert "mock_mode" in settings
    assert "connection_status" in settings
    assert "baudrate" in settings
    assert "available_ports" in settings
    assert isinstance(settings["available_ports"], list)


def test_post_serial_settings(client: TestClient):
    """Verifies updating serial settings via POST /api/settings/serial."""
    response = client.post(
        "/api/settings/serial",
        json={
            "mock_mode": True,
            "baudrate": 115200,
            "sim_heart_rate": 78.0,
            "sim_leads_off": 0,
            "sim_finger_contact": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mock_mode"] is True
    assert data["baudrate"] == 115200
    assert data["sim_heart_rate"] == 78.0


def test_simulator_controls_endpoint(client: TestClient):
    """Verifies POST /api/settings/simulator updates simulator parameters."""
    response = client.post(
        "/api/settings/simulator",
        json={"heart_rate": 84.0, "leads_off": 0, "finger_contact": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    assert data.get("sim_heart_rate") == 84.0


def test_system_status_endpoint(client: TestClient):
    """Verifies GET /api/status returns overall system and hardware telemetry state."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "serial" in data
    assert data["serial"]["mock_mode"] is True


# ============================================================================
# 5. Telemetry WebSocket Endpoint (/ws/telemetry)
# ============================================================================

def test_telemetry_websocket_connection_and_init(client: TestClient):
    """Verifies WebSocket client connection and initial handshake state packet."""
    with client.websocket_connect("/ws/telemetry") as ws:
        packet = ws.receive_json()
        assert packet.get("type") in ("init", "telemetry")
        if packet.get("type") == "init":
            assert "session" in packet
            assert "serial_status" in packet


def test_telemetry_websocket_bidirectional_ping(client: TestClient):
    """Verifies sending ping command to /ws/telemetry yields a pong response."""
    with client.websocket_connect("/ws/telemetry") as ws:
        # First read initial or incoming frame
        ws.receive_json()

        # Send ping
        ws.send_json({"command": "ping"})

        # Read until pong is received (or max 5 iterations to allow broadcaster frames)
        received_types = []
        for _ in range(5):
            msg = ws.receive_json()
            t = msg.get("type")
            received_types.append(t)
            if t == "pong":
                break
        assert "pong" in received_types


def test_telemetry_websocket_sim_control_command(client: TestClient):
    """Verifies modifying simulator controls over the WebSocket channel."""
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.receive_json()

        # Send set_sim_controls command
        ws.send_json({
            "command": "set_sim_controls",
            "heart_rate": 90.0,
            "leads_off": 0,
            "finger_contact": True
        })

        # Read for confirmation or broadcast
        received = []
        for _ in range(5):
            msg = ws.receive_json()
            received.append(msg.get("type"))
            if "sim_controls_updated" in received:
                break
        assert "sim_controls_updated" in received


# ============================================================================
# 6. Face Processing & Enrollment Endpoints
# ============================================================================

def test_face_process_endpoint(client: TestClient):
    """Verifies POST /api/face/process accepts frames and returns detection structures."""
    b64_face = generate_synthetic_face_base64()
    response = client.post(
        "/api/face/process",
        json={"image": b64_face, "recognition_thresh": 0.40, "return_annotated": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "count" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_face_process_invalid_image(client: TestClient):
    """Verifies POST /api/face/process rejects invalid image data with HTTP 400."""
    response = client.post(
        "/api/face/process",
        json={"image": "invalid_base64_string", "recognition_thresh": 0.40}
    )
    assert response.status_code == 400


def test_face_enroll_endpoint(client: TestClient):
    """Verifies POST /api/face/enroll structure validation."""
    b64_face = generate_synthetic_face_base64()
    response = client.post(
        "/api/face/enroll",
        json={
            "name": f"Cadet_{int(time.time())}",
            "image": b64_face,
            "notes": "Flight Simulation Unit"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "message" in data


# ============================================================================
# 7. Data Export Endpoints (CSV timeseries & metrics)
# ============================================================================

def test_data_export_timeseries_and_metrics_csv(client: TestClient):
    """Verifies timeseries and metrics CSV export endpoints."""
    export_sess_id = f"EXPORT_SES_{int(time.time())}"
    state.db.create_session(export_sess_id, is_guest=True)

    # Insert test samples and metrics
    state.db.insert_sensor_samples([
        (export_sess_id, 1000, 1, 512, 60000, 75000, 0),
        (export_sess_id, 1010, 2, 520, 60100, 75100, 0),
    ])
    state.db.insert_metric(export_sess_id, {
        "timestamp_s": 1000.0,
        "ecg_hr": 72.0,
        "ppg_hr": 72.0,
        "spo2": 98.0,
    })

    # Test timeseries CSV export
    ts_resp = client.get(f"/api/reports/export/timeseries/{export_sess_id}")
    assert ts_resp.status_code == 200
    assert "text/csv" in ts_resp.headers.get("content-type", "")
    assert "timestamp_ms" in ts_resp.text

    # Test metrics CSV export
    m_resp = client.get(f"/api/reports/export/metrics/{export_sess_id}")
    assert m_resp.status_code == 200
    assert "text/csv" in m_resp.headers.get("content-type", "")
    assert "timestamp_s" in m_resp.text

    # Cleanup
    state.db.delete_session(export_sess_id)
