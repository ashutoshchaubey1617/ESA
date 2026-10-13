"""
Unified Multimodal Health & Access System Dashboard
Streamlit-based application integrating Face Recognition, Arduino Uno R3 telemetry,
biomedical signal processing (ECG, PPG, SpO2, HRV, PAT, Respiration),
machine learning blood pressure analysis, and SQLite session persistence.
"""

import io
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from biomedical.combined import CombinedBiomedicalAnalyzer
from biomedical.ecg import ECGProcessor
from biomedical.ppg import PPGProcessor
from database.db import DatabaseManager
from face_pipeline.face_service import FaceService
from hardware.serial_reader import SerialManager
from ml.bp_model import BloodPressureFeatureExtractor, BloodPressureModel

# Page configuration
st.set_page_config(
    page_title="Multimodal Health & Access System",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Clinical & Diagnostic Dashboard
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .alert-lead-off {
        background-color: rgba(230, 57, 70, 0.15);
        border-left: 5px solid #e63946;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 12px;
        color: #ff6b6b;
        font-weight: 600;
    }
    .alert-finger-off {
        background-color: rgba(244, 162, 97, 0.15);
        border-left: 5px solid #f4a261;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 12px;
        color: #f4a261;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize persistent session state components
if "db" not in st.session_state:
    st.session_state.db = DatabaseManager("health_monitor.db")

if "serial_manager" not in st.session_state:
    st.session_state.serial_manager = SerialManager(port=None, baudrate=115200, mock_mode=True)
    st.session_state.serial_manager.start()

if "face_service" not in st.session_state:
    st.session_state.face_service = FaceService(st.session_state.db)

if "ecg_proc" not in st.session_state:
    st.session_state.ecg_proc = ECGProcessor(sampling_rate=100.0)

if "ppg_proc" not in st.session_state:
    st.session_state.ppg_proc = PPGProcessor(sampling_rate=100.0)

if "combined_proc" not in st.session_state:
    st.session_state.combined_proc = CombinedBiomedicalAnalyzer(sampling_rate=100.0)

if "bp_model" not in st.session_state:
    st.session_state.bp_model = BloodPressureModel()

if "bp_extractor" not in st.session_state:
    st.session_state.bp_extractor = BloodPressureFeatureExtractor(sampling_rate=100.0)

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if "active_participant" not in st.session_state:
    st.session_state.active_participant = None

if "is_guest_session" not in st.session_state:
    st.session_state.is_guest_session = False

db: DatabaseManager = st.session_state.db
serial_mgr: SerialManager = st.session_state.serial_manager
face_srv: FaceService = st.session_state.face_service
ecg_proc: ECGProcessor = st.session_state.ecg_proc
ppg_proc: PPGProcessor = st.session_state.ppg_proc
combined_proc: CombinedBiomedicalAnalyzer = st.session_state.combined_proc
bp_model: BloodPressureModel = st.session_state.bp_model
bp_extractor: BloodPressureFeatureExtractor = st.session_state.bp_extractor


# Helper function to acquire and process the latest window of samples
def get_current_biomedical_state():
    samples = serial_mgr.get_latest_samples(600)  # 6-second sliding window at 100 Hz
    if not samples or len(samples) < 150:
        return None

    # Persist samples to DB if session is active
    if st.session_state.active_session_id:
        to_persist = [(st.session_state.active_session_id, s[0], s[1], s[2], s[3], s[4], s[5]) for s in samples[-10:]]
        db.insert_sensor_samples(to_persist)

    arr = np.array(samples)
    timestamps = arr[:, 0]
    ecg_raw = arr[:, 2]
    ppg_red = arr[:, 3]
    ppg_ir = arr[:, 4]
    leads_off = int(arr[-1, 5])

    ecg_res = ecg_proc.process(ecg_raw, leads_off=leads_off)
    ppg_res = ppg_proc.process(ppg_red, ppg_ir)

    ecg_hr = ecg_res["heart_rate"]
    ppg_hr = ppg_res["heart_rate"]

    hr_comp = combined_proc.compare_heart_rates(ecg_hr, ppg_hr)
    pat_res = combined_proc.compute_pulse_arrival_time(ecg_res["r_peaks"], ppg_res["pulse_peaks"])
    resp_res = combined_proc.estimate_respiratory_rate(ecg_res["filtered_ecg"], ecg_res["r_peaks"], ppg_res["filtered_ir"])

    # Optional ML BP estimation
    bp_features = bp_extractor.extract_features(
        ecg_res["r_peaks"], ecg_res["filtered_ecg"], ppg_res["pulse_peaks"], ppg_res["filtered_ir"], ppg_ir
    )
    bp_res = (
        bp_model.predict(bp_features)
        if bp_features
        else {
            "sbp_mmHg": None,
            "dbp_mmHg": None,
            "status": "UNCALIBRATED",
            "disclaimer": "Blood Pressure estimation is research-grade and disabled until individual cuff calibration.",
        }
    )

    # Periodic metrics persistence
    if st.session_state.active_session_id and len(samples) % 50 == 0:
        db.insert_metric(
            st.session_state.active_session_id,
            {
                "timestamp_s": time.time(),
                "ecg_hr": ecg_hr,
                "ppg_hr": ppg_hr,
                "spo2": ppg_res["spo2"],
                "rr_interval_ms": ecg_res["rr_intervals_ms"][-1] if len(ecg_res["rr_intervals_ms"]) > 0 else None,
                "rmssd_ms": ecg_res["hrv"].get("rmssd_ms"),
                "sdnn_ms": ecg_res["hrv"].get("sdnn_ms"),
                "pat_ms": pat_res.get("pat_median_ms"),
                "resp_rate_bpm": resp_res.get("respiratory_rate_bpm"),
                "signal_quality": ecg_res["signal_quality"]["quality"],
                "bp_systolic": bp_res.get("sbp_mmHg"),
                "bp_diastolic": bp_res.get("dbp_mmHg"),
            },
        )

    is_lead_off = bool(leads_off == 1 or ecg_raw[-1] >= 1000 or ecg_res["signal_quality"]["quality"] == "DISCONNECTED")
    is_finger_off = bool(not ppg_res["contact_status"]["contact"])

    return {
        "timestamps": timestamps,
        "ecg_raw": ecg_raw,
        "ppg_red": ppg_red,
        "ppg_ir": ppg_ir,
        "ecg": ecg_res,
        "ppg": ppg_res,
        "hr_comp": hr_comp,
        "pat": pat_res,
        "resp": resp_res,
        "bp": bp_res,
        "is_lead_off": is_lead_off,
        "is_finger_off": is_finger_off,
    }


# ----------------- PARTICIPANT ENROLLMENT MODAL -----------------
@st.dialog("➕ Enroll New Participant", width="large")
def show_enrollment_dialog(prefill_image: Optional[np.ndarray] = None):
    st.markdown("Register participant identity into the biometric database with facial landmark extraction.")
    name = st.text_input("Full Name / Subject ID", key="modal_reg_name")
    notes = st.text_area("Clinical Notes / Subject Metadata (Optional)", key="modal_reg_notes")

    col_cam, col_file = st.columns(2)
    with col_cam:
        modal_cam = st.camera_input("Capture Face from Webcam", key="modal_cam_snap")
    with col_file:
        modal_file = st.file_uploader("Or Upload Face Image", type=["jpg", "jpeg", "png"], key="modal_file_snap")

    img_src = modal_cam or modal_file
    if img_src is not None:
        cv2_img = cv2.imdecode(np.frombuffer(img_src.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    elif prefill_image is not None:
        cv2_img = prefill_image
    else:
        cv2_img = None

    if cv2_img is not None:
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="Biometric Face Sample", width=240)

    if st.button("Complete Enrollment", type="primary", use_container_width=True):
        if not name or not name.strip():
            st.error("Please enter a valid participant name or identifier.")
        elif cv2_img is None:
            st.error("Please capture or upload a clear facial image.")
        else:
            with st.spinner("Extracting facial embeddings & synchronizing database..."):
                res = face_srv.enroll_participant(name.strip(), cv2_img, notes=notes.strip())
            if res["success"]:
                st.success(f"✅ {res['message']}")
                time.sleep(0.8)
                st.rerun()
            else:
                st.error(f"❌ {res['message']}")


# ----------------- SIDEBAR CONTROLS & STATUS -----------------
st.sidebar.title("💓 System Hub")

# Active User & Session Box
st.sidebar.subheader("Active Session")
if st.session_state.active_session_id:
    participant_label = (
        st.session_state.active_participant.get("name")
        if st.session_state.active_participant
        else ("Guest User" if st.session_state.is_guest_session else "Unassigned")
    )
    st.sidebar.success(f"🟢 Active: `{st.session_state.active_session_id}`")
    st.sidebar.info(f"👤 Participant: **{participant_label}**")

    if st.sidebar.button("⏹️ End Current Session", use_container_width=True):
        db.end_session(st.session_state.active_session_id)
        st.session_state.active_session_id = None
        st.session_state.active_participant = None
        st.session_state.is_guest_session = False
        st.rerun()
else:
    st.sidebar.warning("⚪ No active session")
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("New Guest Session", use_container_width=True):
            new_id = f"GUEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db.create_session(new_id, participant_id=None, is_guest=True)
            st.session_state.active_session_id = new_id
            st.session_state.is_guest_session = True
            st.session_state.active_participant = None
            st.rerun()
    with col_s2:
        enrolled_users = db.get_all_participants()
        if enrolled_users:
            selected_user_name = st.selectbox(
                "Select User", [u["name"] for u in enrolled_users], label_visibility="collapsed"
            )
            if st.button("Start User Session", use_container_width=True):
                user_obj = db.get_participant_by_name(selected_user_name)
                new_id = f"SESS_{user_obj['participant_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                db.create_session(new_id, participant_id=user_obj["participant_id"], is_guest=False)
                st.session_state.active_session_id = new_id
                st.session_state.active_participant = user_obj
                st.session_state.is_guest_session = False
                st.rerun()

st.sidebar.divider()

# Hardware Connection Status
st.sidebar.subheader("Hardware Status")
if serial_mgr.mock_mode:
    st.sidebar.warning("🧪 Mode: SIMULATED (Synthetic 100Hz)")
else:
    if serial_mgr.connection_status == "CONNECTED":
        st.sidebar.success(f"🔌 Connected: {serial_mgr.port} @ {serial_mgr.baudrate}")
    elif serial_mgr.connection_status == "CONNECTING":
        st.sidebar.info("⏳ Connecting to Arduino...")
    else:
        st.sidebar.error(f"❌ Serial: {serial_mgr.connection_status}")
        if serial_mgr.last_error:
            st.sidebar.caption(serial_mgr.last_error)

# Camera / Face Recog Status
if face_srv.is_ready:
    st.sidebar.success("📷 Face Recognition: Ready")
else:
    st.sidebar.warning(f"📷 Face Recog: {face_srv.status_message}")

# Navigation Menu
nav = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview Dashboard",
        "2. Face Recognition & Access",
        "3. Live ECG",
        "4. Live PPG & SpO2",
        "5. Health Analytics",
        "6. Session History",
        "7. Reports & Export",
        "8. Settings",
    ],
)


# ----------------- 1. OVERVIEW DASHBOARD -----------------
if nav == "1. Overview Dashboard":
    st.header("💓 Overview Clinical Dashboard")

    state = get_current_biomedical_state()

    # Active Session Indicator Banner
    if st.session_state.active_session_id:
        p_name = st.session_state.active_participant["name"] if st.session_state.active_participant else "Guest User"
        st.caption(f"🟢 **Monitoring Session Active:** `{st.session_state.active_session_id}` | Participant: **{p_name}**")
    else:
        st.caption("⚪ **Unassigned Telemetry Stream** (No active session bound. Start or bind session via sidebar or Face Access).")

    # Real-Time Diagnostic Alert Banners
    if state:
        if state["is_lead_off"]:
            st.error(
                "🚨 **REAL-TIME ALERT: AD8232 ECG Lead-Off Detected!** Electrodes are detached (LOD+/LOD- tripped or railed ADC). Check RA, LA, RL electrode placement."
            )
        if state["is_finger_off"]:
            st.warning(
                "⚠️ **REAL-TIME ALERT: MAX30102 Finger-Off Detected!** Optical contact lost (< 10k counts). Place finger firmly on sensor."
            )
        if not state["is_lead_off"] and not state["is_finger_off"] and not state["hr_comp"]["is_concordant"] and state["hr_comp"]["difference_bpm"] is not None:
            st.info(
                f"ℹ️ **Physiological HR Divergence:** ECG ({state['ecg']['heart_rate']} BPM) vs PPG ({state['ppg']['heart_rate']} BPM) differ by {state['hr_comp']['difference_bpm']} BPM. Inspect for motion artifacts."
            )

    # 8 Key Vital Metrics (2 Rows x 4 Columns)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hr_val = f"{state['ecg']['heart_rate']} BPM" if state and state["ecg"]["heart_rate"] else "N/A"
        st.metric("ECG Heart Rate", hr_val)
    with c2:
        ppg_val = f"{state['ppg']['heart_rate']} BPM" if state and state["ppg"]["heart_rate"] else "N/A"
        st.metric("PPG Heart Rate", ppg_val)
    with c3:
        spo2_val = f"{state['ppg']['spo2']} %" if state and state["ppg"]["spo2"] else "N/A"
        st.metric("SpO2 Saturation", spo2_val)
    with c4:
        pat_val = f"{state['pat']['pat_median_ms']} ms" if state and state["pat"]["pat_median_ms"] else "N/A"
        st.metric("Pulse Arrival Time (PAT)", pat_val)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        rmssd_val = f"{state['ecg']['hrv']['rmssd_ms']} ms" if state and state["ecg"]["hrv"]["rmssd_ms"] else "N/A"
        st.metric("HRV (RMSSD)", rmssd_val)
    with c6:
        sdnn_val = f"{state['ecg']['hrv']['sdnn_ms']} ms" if state and state["ecg"]["hrv"]["sdnn_ms"] else "N/A"
        st.metric("HRV (SDNN)", sdnn_val)
    with c7:
        rr_val = f"{state['resp']['respiratory_rate_bpm']} BrPM" if state and state["resp"]["respiratory_rate_bpm"] else "N/A"
        st.metric("Respiration Rate", rr_val)
    with c8:
        quality_label = state["ecg"]["signal_quality"]["quality"] if state else "NO_SIGNAL"
        st.metric("Signal Quality", quality_label)

    # Dual Waveform Scopes
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.subheader("ECG Real-Time Scope (Lead II)")
        if state and len(state["ecg"]["filtered_ecg"]) > 0:
            fig, ax = plt.subplots(figsize=(7, 2.7))
            filtered_ecg = state["ecg"]["filtered_ecg"][-300:]
            t_ecg = np.arange(len(filtered_ecg)) / 100.0
            ax.plot(t_ecg, filtered_ecg, color="#e63946", lw=1.6, label="Filtered ECG")
            ax.set_title("Filtered ECG Waveform (0.5 - 40 Hz)", fontsize=10)
            ax.set_xlabel("Time Window (seconds)", fontsize=9)
            ax.set_ylabel("Amplitude (mV)", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Awaiting ECG stream...")

    with col_w2:
        st.subheader("PPG Real-Time Scope (IR & Red)")
        if state and len(state["ppg"]["filtered_ir"]) > 0:
            fig, ax = plt.subplots(figsize=(7, 2.7))
            filtered_ir = state["ppg"]["filtered_ir"][-300:]
            filtered_red = state["ppg"]["filtered_red"][-300:]
            t_ppg = np.arange(len(filtered_ir)) / 100.0
            ax.plot(t_ppg, filtered_ir, color="#0077b6", lw=1.6, label="IR (880nm)")
            ax.plot(t_ppg, filtered_red, color="#e76f51", lw=1.2, alpha=0.8, label="Red (660nm)")
            ax.set_title("Filtered PPG Pulsatile Components", fontsize=10)
            ax.set_xlabel("Time Window (seconds)", fontsize=9)
            ax.set_ylabel("AC Amplitude", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Awaiting PPG stream...")

    # Real-time refresh loop button
    if st.checkbox("Live Continuous Monitor (Auto-Refresh)", value=True):
        time.sleep(0.5)
        st.rerun()


# ----------------- 2. FACE RECOGNITION & ACCESS -----------------
elif nav == "2. Face Recognition & Access":
    st.header("👤 Face Recognition, Tracking & Biometric Access")

    tab_recog, tab_enroll, tab_mgmt = st.tabs(["Live Recognition & Access", "Participant Enrollment", "Enrolled Directory"])

    with tab_recog:
        st.subheader("Webcam Face Verification Stream")
        img_file = st.camera_input("Capture Face for Verification & Auto-Session Binding")

        if img_file is not None:
            bytes_data = img_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            res = face_srv.process_frame(cv2_img)
            st.image(cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB), caption="Processed Face Frame", use_container_width=True)

            if res["detections"]:
                for idx, d in enumerate(res["detections"]):
                    with st.container(border=True):
                        st.markdown(
                            f"**Subject:** `{d['name']}` | **Match Similarity:** `{d['similarity']:.2f}` | "
                            f"**Liveness:** `{d['liveness']['status']}` (Confidence: {d['liveness']['score']:.2f})"
                        )

                        if d["is_known"] and d["participant_id"]:
                            is_already_active = (
                                st.session_state.active_session_id is not None
                                and st.session_state.active_participant is not None
                                and st.session_state.active_participant.get("participant_id") == d["participant_id"]
                            )

                            if is_already_active:
                                st.success(
                                    f"🟢 Monitoring Session is currently active and bound to **{d['name']}** (`{st.session_state.active_session_id}`)."
                                )
                            else:
                                btn_label = f"🔗 Bind Monitoring Session to {d['name']}"
                                if st.session_state.active_session_id:
                                    btn_label = f"🔄 Switch Active Session to {d['name']}"

                                if st.button(btn_label, key=f"bind_{d['name']}_{idx}", type="primary"):
                                    if st.session_state.active_session_id:
                                        db.end_session(st.session_state.active_session_id)
                                    participant_obj = db.get_participant_by_name(d["name"])
                                    new_id = f"SESS_{d['participant_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    db.create_session(new_id, participant_id=d["participant_id"], is_guest=False)
                                    st.session_state.active_session_id = new_id
                                    st.session_state.active_participant = participant_obj
                                    st.session_state.is_guest_session = False
                                    st.success(f"Session bound successfully to {d['name']}!")
                                    st.rerun()
                        else:
                            st.warning("⚠️ Unrecognized Subject (Not found in enrolled database).")
                            col_u1, col_u2 = st.columns(2)
                            with col_u1:
                                if st.button("➕ Quick Enroll Detected Face (Modal)", key=f"quick_enr_{idx}"):
                                    show_enrollment_dialog(prefill_image=cv2_img)
                            with col_u2:
                                if st.button("👤 Start Guest Session for Subject", key=f"guest_unk_{idx}"):
                                    if st.session_state.active_session_id:
                                        db.end_session(st.session_state.active_session_id)
                                    new_id = f"GUEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    db.create_session(new_id, participant_id=None, is_guest=True)
                                    st.session_state.active_session_id = new_id
                                    st.session_state.active_participant = None
                                    st.session_state.is_guest_session = True
                                    st.success("Guest session started!")
                                    st.rerun()
            else:
                st.warning("No face detected in camera capture. Ensure face is well-lit and facing forward.")

    with tab_enroll:
        st.subheader("Enroll New Participant")
        if st.button("➕ Launch Interactive Enrollment Modal Dialog", type="primary"):
            show_enrollment_dialog()

        st.divider()
        st.markdown("#### Direct Inline Enrollment")
        enroll_name = st.text_input("Participant Full Name / Subject ID", key="inline_enroll_name")
        enroll_notes = st.text_area("Clinical / Participant Notes (Optional)", key="inline_enroll_notes")
        enroll_img_file = st.file_uploader("Upload Clear Face Photo", type=["jpg", "jpeg", "png"], key="inline_enroll_file")
        enroll_cam = st.camera_input("Or Capture Face from Camera", key="inline_enroll_cam")

        active_enroll_img = enroll_cam or enroll_img_file

        if st.button("Complete Inline Enrollment", key="btn_inline_enr") and enroll_name and active_enroll_img:
            bytes_data = active_enroll_img.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            res = face_srv.enroll_participant(enroll_name.strip(), cv2_img, notes=enroll_notes.strip())
            if res["success"]:
                st.success(res["message"])
                time.sleep(0.8)
                st.rerun()
            else:
                st.error(res["message"])

    with tab_mgmt:
        st.subheader("Registered Participants Database")
        participants = db.get_all_participants()
        if participants:
            df_part = pd.DataFrame(
                [
                    {
                        "ID": p["participant_id"],
                        "Name": p["name"],
                        "Enrolled At": p["enrolled_at"],
                        "Has Embedding": p.get("embedding") is not None,
                        "Notes": p.get("notes", ""),
                    }
                    for p in participants
                ]
            )
            st.dataframe(df_part, use_container_width=True)

            del_target = st.selectbox("Select Participant to Delete", [p["name"] for p in participants])
            if st.button("🗑️ Delete Participant Record"):
                part_obj = db.get_participant_by_name(del_target)
                if part_obj:
                    db.delete_participant(part_obj["participant_id"])
                    face_srv.reload_known_faces()
                    st.success(f"Deleted {del_target}")
                    st.rerun()
        else:
            st.info("No participants enrolled yet.")


# ----------------- 3. LIVE ECG -----------------
elif nav == "3. Live ECG":
    st.header("📈 Live ECG Oscilloscope (AD8232)")

    state = get_current_biomedical_state()
    if not state:
        st.info("Waiting for ECG stream...")
    else:
        ecg_info = state["ecg"]
        sqi = ecg_info["signal_quality"]

        # Prominent Real-time Lead-Off Alert Banner
        if state["is_lead_off"]:
            st.error(
                "🚨 **REAL-TIME ALERT: AD8232 Lead-Off Active!** Electrodes are detached (LOD+/LOD- tripped or railed ADC). Check RA, LA, RL electrode placement."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ECG Heart Rate", f"{ecg_info['heart_rate']} BPM" if ecg_info["heart_rate"] else "N/A")
        m2.metric("Detected R-Peaks", len(ecg_info["r_peaks"]))
        m3.metric("Signal Quality", sqi["quality"], delta=f"SNR: {sqi['snr_db']} dB")
        m4.metric("Hardware Lead Status", "LEAD OFF" if state["is_lead_off"] else "NORMAL")

        show_raw = st.checkbox("Overlay Raw ADC (0-1023)", value=False)

        fig, ax = plt.subplots(figsize=(12, 4))
        filtered = ecg_info["filtered_ecg"]
        t_axis = np.arange(len(filtered)) / 100.0  # seconds

        ax.plot(t_axis, filtered, color="#d90429", lw=1.6, label="Filtered ECG (0.5 - 40 Hz)")

        # Mark R-peaks
        if len(ecg_info["r_peaks"]) > 0:
            valid_peaks = ecg_info["r_peaks"][ecg_info["r_peaks"] < len(filtered)]
            ax.scatter(valid_peaks / 100.0, filtered[valid_peaks], color="#003049", s=60, zorder=5, label="R-Peak")

        if show_raw:
            ax2 = ax.twinx()
            ax2.plot(t_axis, state["ecg_raw"], color="#8d99ae", lw=0.8, alpha=0.5, label="Raw ADC")
            ax2.set_ylabel("Raw ADC Units (10-bit)")

        ax.set_title("ECG Waveform & Detected R-Peaks", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Beat-to-Beat R-R intervals
        if len(ecg_info["rr_intervals_ms"]) > 0:
            st.subheader("Beat-to-Beat R-R Intervals (ms)")
            rr_arr = ecg_info["rr_intervals_ms"]
            df_rr = pd.DataFrame({"Beat Index": range(1, len(rr_arr) + 1), "RR_Interval_ms": rr_arr})
            st.line_chart(df_rr.set_index("Beat Index"), use_container_width=True)
            col_rr1, col_rr2, col_rr3 = st.columns(3)
            col_rr1.metric("Mean R-R", f"{float(np.mean(rr_arr)):.1f} ms")
            col_rr2.metric("Min R-R", f"{float(np.min(rr_arr)):.1f} ms")
            col_rr3.metric("Max R-R", f"{float(np.max(rr_arr)):.1f} ms")

    if st.checkbox("Continuous Refresh", value=True):
        time.sleep(0.5)
        st.rerun()


# ----------------- 4. LIVE PPG & SPO2 -----------------
elif nav == "4. Live PPG & SpO2":
    st.header("🩸 Live PPG & Pulse Oximetry (MAX30102)")

    state = get_current_biomedical_state()
    if not state:
        st.info("Waiting for PPG stream...")
    else:
        ppg_info = state["ppg"]
        contact = ppg_info["contact_status"]

        # Prominent Real-time Finger-Off Alert Banner
        if state["is_finger_off"]:
            st.warning(
                "⚠️ **REAL-TIME ALERT: MAX30102 Finger-Off Active!** Optical DC return is below threshold (< 10,000 counts). Plethysmogram and SpO2 calculation paused."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PPG Pulse Rate", f"{ppg_info['heart_rate']} BPM" if ppg_info["heart_rate"] else "N/A")
        m2.metric("SpO2 Saturation", f"{ppg_info['spo2']} %" if ppg_info["spo2"] else "N/A")
        m3.metric("Finger Contact", "DETECTED" if contact["contact"] else "NO FINGER")
        m4.metric("Optical Return (IR DC)", f"{int(contact['ir_dc'])} counts")

        fig, ax = plt.subplots(figsize=(12, 4))
        filtered_ir = ppg_info["filtered_ir"]
        t_axis = np.arange(len(filtered_ir)) / 100.0

        ax.plot(t_axis, filtered_ir, color="#0077b6", lw=1.8, label="Infrared (880nm)")
        ax.plot(t_axis, ppg_info["filtered_red"], color="#e63946", lw=1.2, alpha=0.7, label="Red (660nm)")

        if len(ppg_info["pulse_peaks"]) > 0:
            valid_p = ppg_info["pulse_peaks"][ppg_info["pulse_peaks"] < len(filtered_ir)]
            ax.scatter(valid_p / 100.0, filtered_ir[valid_p], color="#d62828", s=50, zorder=5, label="Systolic Peak")

        ax.set_title("Dual-Wavelength PPG Waveforms & Systolic Peaks", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("AC Pulsatile Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # SpO2 Clinical Category
        if ppg_info["spo2"]:
            spo2_val = ppg_info["spo2"]
            if spo2_val >= 95.0:
                st.success(f"🟢 SpO2 Level: {spo2_val}% (Normal Oxygenation)")
            elif spo2_val >= 90.0:
                st.warning(f"🟡 SpO2 Level: {spo2_val}% (Mild Hypoxemia)")
            else:
                st.error(f"🔴 SpO2 Level: {spo2_val}% (Moderate/Severe Hypoxemia)")

    if st.checkbox("Continuous Refresh", value=True):
        time.sleep(0.5)
        st.rerun()


# ----------------- 5. HEALTH ANALYTICS -----------------
elif nav == "5. Health Analytics":
    st.header("🔬 Advanced Biomedical Analytics & Machine Learning")

    state = get_current_biomedical_state()
    if not state:
        st.info("Insufficient sensor data for analytics. Ensure hardware telemetry is active.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                st.subheader("ECG vs PPG Concordance")
                comp = state["hr_comp"]
                st.write(f"**Concordance Status:** `{comp['status']}`")
                st.write(f"**Absolute Difference:** `{comp['difference_bpm']} BPM`")
                if comp["is_concordant"]:
                    st.success("✅ Cardiac modalities concordant. Minimal motion interference detected.")
                elif comp["difference_bpm"] is not None:
                    st.warning("⚠️ HR discrepancy detected between electrical (ECG) and optical (PPG) modalities.")

            with st.container(border=True):
                st.subheader("Pulse Arrival Time (PAT)")
                pat = state["pat"]
                st.write(f"**Median PAT:** `{pat['pat_median_ms']} ms` (Standard Dev: {pat['pat_std_ms']} ms)")
                st.write(f"**Matched Cardiac Cycles:** `{pat['valid_pairs_count']}`")
                st.caption(
                    "Note: PAT measures R-peak to peripheral pulse arrival ($PAT = PEP + PTT$). "
                    "It differs from true arterial PTT by including the pre-ejection period (PEP)."
                )

        with c2:
            with st.container(border=True):
                st.subheader("Respiratory Rate Analysis")
                resp = state["resp"]
                st.write(f"**Estimated Rate:** `{resp['respiratory_rate_bpm']} Breaths/min`")
                st.write(f"**Extraction Modality:** `{resp['method']}`")
                st.write(f"**Diagnostic Status:** `{resp['status']}`")

            with st.container(border=True):
                st.subheader("Cuffless Blood Pressure Estimation (Random Forest)")
                bp = state["bp"]
                st.write(f"**Systolic BP (SBP):** `{bp['sbp_mmHg']} mmHg`")
                st.write(f"**Diastolic BP (DBP):** `{bp['dbp_mmHg']} mmHg`")
                st.write(f"**Model Status:** `{bp['status']}`")

                if bp.get("sbp_mmHg") and bp.get("dbp_mmHg"):
                    sbp_val = bp["sbp_mmHg"]
                    dbp_val = bp["dbp_mmHg"]
                    if sbp_val < 120 and dbp_val < 80:
                        st.success("🟢 Category: Normal Blood Pressure")
                    elif 120 <= sbp_val < 130 and dbp_val < 80:
                        st.info("🟡 Category: Elevated Blood Pressure")
                    elif 130 <= sbp_val < 140 or 80 <= dbp_val < 90:
                        st.warning("🟠 Category: Stage 1 Hypertension")
                    else:
                        st.error("🔴 Category: Stage 2 Hypertension")

                if not bp_model.is_trained:
                    if st.button("🧪 Train Baseline Research Calibration (Synthetic Cohort)"):
                        np.random.seed(42)
                        n_samples = 200
                        pats = np.random.uniform(150, 300, n_samples)
                        hrs = np.random.uniform(60, 100, n_samples)
                        rises = np.random.uniform(80, 150, n_samples)
                        widths = np.random.uniform(180, 250, n_samples)
                        pis = np.random.uniform(1.0, 3.0, n_samples)
                        y_sbps = 180.0 - 0.25 * pats + 0.2 * hrs + np.random.normal(0, 3, n_samples)
                        y_dbps = 110.0 - 0.15 * pats + 0.1 * hrs + np.random.normal(0, 2, n_samples)
                        synth_data = [
                            {
                                "pat_ms": pats[i],
                                "hr_bpm": hrs[i],
                                "rise_time_ms": rises[i],
                                "pulse_width_ms": widths[i],
                                "perfusion_index": pis[i],
                                "sbp": y_sbps[i],
                                "dbp": y_dbps[i],
                            }
                            for i in range(n_samples)
                        ]
                        eval_res = bp_model.train(synth_data)
                        st.success(
                            f"Model calibrated! SBP MAE: {eval_res['sbp']['mae_mmHg']} mmHg | DBP MAE: {eval_res['dbp']['mae_mmHg']} mmHg"
                        )
                        st.rerun()

                st.info(f"**Clinical Disclaimer:** {bp['disclaimer']}")


# ----------------- 6. SESSION HISTORY -----------------
elif nav == "6. Session History":
    st.header("🗄️ Monitoring Session History")

    all_sessions = db.get_all_sessions()
    if not all_sessions:
        st.info("No recorded sessions found in database.")
    else:
        df_sess = pd.DataFrame(
            [
                {
                    "Session ID": s["session_id"],
                    "Participant": s.get("participant_name") or ("Guest" if s["is_guest"] else "Unknown"),
                    "Start Time": s["start_time"],
                    "End Time": s["end_time"] or "IN_PROGRESS",
                    "Duration (s)": round(s["duration_seconds"] or 0, 1),
                    "Status": s["status"],
                }
                for s in all_sessions
            ]
        )
        st.dataframe(df_sess, use_container_width=True)

        selected_sid = st.selectbox("Inspect Session Details", [s["session_id"] for s in all_sessions])
        if selected_sid:
            sess_meta = db.get_session(selected_sid)
            if sess_meta:
                c_s1, c_s2, c_s3, c_s4 = st.columns(4)
                c_s1.metric("Subject", sess_meta.get("participant_name") or "Guest")
                c_s2.metric("Status", sess_meta["status"])
                c_s3.metric("Duration", f"{round(sess_meta.get('duration_seconds') or 0, 1)}s")
                c_s4.metric("Start Time", sess_meta["start_time"][:19] if sess_meta.get("start_time") else "N/A")

            metrics = db.get_session_metrics(selected_sid)
            if metrics:
                df_m = pd.DataFrame(metrics)
                st.subheader("Physiological Metrics Progression")
                st.line_chart(df_m.set_index("timestamp_s")[["ecg_hr", "ppg_hr", "spo2"]], use_container_width=True)

                if "bp_systolic" in df_m.columns and df_m["bp_systolic"].notna().any():
                    st.subheader("Blood Pressure Progression")
                    st.line_chart(df_m.set_index("timestamp_s")[["bp_systolic", "bp_diastolic"]], use_container_width=True)
            else:
                st.info("No computed metrics logged for this session.")

            if st.button("🗑️ Delete Session", key="del_sess_btn"):
                db.delete_session(selected_sid)
                st.success(f"Session {selected_sid} deleted.")
                st.rerun()


# ----------------- 7. REPORTS & EXPORT -----------------
elif nav == "7. Reports & Export":
    st.header("📥 Reports & Clinical Data Export")

    all_sessions = db.get_all_sessions()
    if not all_sessions:
        st.info("No sessions available to export.")
    else:
        export_sid = st.selectbox("Select Session for Export & Reporting", [s["session_id"] for s in all_sessions])
        sess_info = db.get_session(export_sid)
        ts_data = db.get_session_timeseries(export_sid, limit=50000)
        metrics_data = db.get_session_metrics(export_sid)

        tab_csv, tab_report = st.tabs(["CSV Data Downloads", "Clinical Summary Report"])

        with tab_csv:
            if ts_data:
                df_export = pd.DataFrame(ts_data)
                st.subheader(f"Raw Sensor Telemetry Preview ({len(df_export)} samples)")
                st.dataframe(df_export.head(10), use_container_width=True)

                csv_buffer = io.StringIO()
                df_export.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Raw Sensor Timeseries CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{export_sid}_raw_telemetry.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                if metrics_data:
                    df_metrics = pd.DataFrame(metrics_data)
                    csv_m_buf = io.StringIO()
                    df_metrics.to_csv(csv_m_buf, index=False)
                    st.download_button(
                        label="📥 Download Physiological Metrics Summary CSV",
                        data=csv_m_buf.getvalue(),
                        file_name=f"{export_sid}_metrics_summary.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            else:
                st.warning("No timeseries data found for this session.")

        with tab_report:
            st.subheader("Automated Clinical Summary")
            p_name = sess_info.get("participant_name") or ("Guest" if sess_info and sess_info["is_guest"] else "Unknown")
            start_t = sess_info["start_time"] if sess_info else "N/A"
            duration_s = round(sess_info["duration_seconds"] or 0, 1) if sess_info else 0

            avg_hr = "N/A"
            avg_spo2 = "N/A"
            if metrics_data:
                df_m = pd.DataFrame(metrics_data)
                if "ecg_hr" in df_m and df_m["ecg_hr"].notna().any():
                    avg_hr = f"{df_m['ecg_hr'].mean():.1f} BPM"
                if "spo2" in df_m and df_m["spo2"].notna().any():
                    avg_spo2 = f"{df_m['spo2'].mean():.1f} %"

            report_md = f"""### Clinical Telemetry & Access Report
**Session Identifier:** `{export_sid}`  
**Participant Name / ID:** {p_name}  
**Session Start:** {start_t}  
**Session Duration:** {duration_s} seconds  
**Recorded Sensor Samples:** {len(ts_data) if ts_data else 0}  
**Metrics Records Logged:** {len(metrics_data) if metrics_data else 0}  

---
#### Vital Signs Summary
- **Average ECG Heart Rate:** {avg_hr}
- **Average SpO2 Saturation:** {avg_spo2}
- **Telemetry Frequency:** 100 Hz calibrated
- **Electrocardiogram:** Single-lead Lead II (AD8232)
- **Photoplethysmogram:** Red/IR Dual-wavelength (MAX30102)

---
#### Mandatory Clinical Disclaimer
*This system is intended for engineering evaluation, research, and multimodal access control. It is not an FDA-approved medical diagnostic device.*
"""
            st.markdown(report_md)
            st.download_button(
                label="📄 Download Formatted Clinical Report (Markdown)",
                data=report_md,
                file_name=f"Clinical_Report_{export_sid}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ----------------- 8. SETTINGS -----------------
elif nav == "8. Settings":
    st.header("⚙️ System Configuration & Diagnostics")

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        st.subheader("Serial Communication")
        available_ports = serial_mgr.list_available_ports()
        port_options = [p["device"] for p in available_ports]

        selected_port = st.selectbox(
            "Hardware COM Port (Arduino Uno)",
            port_options if port_options else ["None Detected"],
        )
        baud = st.selectbox("Baud Rate", [115200, 57600, 9600], index=0)

        mock_enabled = st.toggle("🧪 Enable Synthetic Mock Hardware Simulator", value=serial_mgr.mock_mode)

        if st.button("Apply Serial Configuration"):
            serial_mgr.stop()
            serial_mgr.port = selected_port if selected_port != "None Detected" else None
            serial_mgr.baudrate = baud
            serial_mgr.mock_mode = mock_enabled
            serial_mgr.start()
            st.success("Serial configuration updated!")
            st.rerun()

        if mock_enabled:
            st.subheader("Simulator Controls")
            sim_hr = st.slider("Simulated Heart Rate (BPM)", 45, 160, int(serial_mgr._sim_heart_rate))
            sim_lead = st.selectbox(
                "Simulated ECG Lead Status",
                [0, 1],
                index=0 if serial_mgr._sim_leads_off == 0 else 1,
                format_func=lambda x: "Normal (Attached)" if x == 0 else "Disconnected (Lead Off)",
            )
            sim_contact = st.toggle("Finger Placed on PPG Sensor", value=serial_mgr._sim_finger_contact)
            serial_mgr.set_simulator_controls(heart_rate=sim_hr, leads_off=sim_lead, finger_contact=sim_contact)

    with col_set2:
        st.subheader("Signal Processing & Models")
        st.write(f"**Sampling Rate:** `100 Hz`")
        st.write(f"**ECG Bandpass:** `0.5 Hz - 40.0 Hz`")
        st.write(f"**PPG Bandpass:** `0.5 Hz - 5.0 Hz`")
        st.write(f"**SpO2 Formula:** `110 - 25 * (AC_red/DC_red)/(AC_ir/DC_ir)`")
        st.write(f"**Face Detector Model:** `SCRFD 2.5G ONNX`")
        st.write(f"**Face Recognizer Model:** `ArcFace iResNet-100 PyTorch`")

        if st.button("Reload Face Embeddings from DB"):
            face_srv.reload_known_faces()
            st.success("Face embeddings synchronized!")

        st.subheader("Database Overview")
        all_p = db.get_all_participants()
        all_s = db.get_all_sessions()
        st.write(f"**Enrolled Participants:** `{len(all_p)}`")
        st.write(f"**Recorded Sessions:** `{len(all_s)}`")
