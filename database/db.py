"""
Database and Persistence Layer for Multimodal Health & Access System
Uses SQLite with thread-safe connections and structured schemas.
"""

import csv
import io
import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class DatabaseManager:
    """Manages SQLite storage for participants, sessions, biometric embeddings, and physiological data."""

    def __init__(self, db_path: str = "health_monitor.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except Exception:
            pass
        return conn

    def init_db(self):
        """Creates tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Participants table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    enrolled_at TEXT NOT NULL,
                    embedding_json TEXT,
                    notes TEXT
                )
                """
            )

            # Sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    participant_id INTEGER,
                    is_guest INTEGER DEFAULT 0,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    status TEXT DEFAULT 'ACTIVE',
                    summary_json TEXT,
                    FOREIGN KEY (participant_id) REFERENCES participants(participant_id) ON DELETE SET NULL
                )
                """
            )

            # Timeseries raw sensor data
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sensor_timeseries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    ecg_raw INTEGER NOT NULL,
                    ppg_red INTEGER NOT NULL,
                    ppg_ir INTEGER NOT NULL,
                    leads_off INTEGER NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )

            # Physiological metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS physiological_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp_s REAL NOT NULL,
                    ecg_hr REAL,
                    ppg_hr REAL,
                    spo2 REAL,
                    rr_interval_ms REAL,
                    rmssd_ms REAL,
                    sdnn_ms REAL,
                    pat_ms REAL,
                    resp_rate_bpm REAL,
                    signal_quality TEXT,
                    bp_systolic REAL,
                    bp_diastolic REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )

            # Indexes for performant timeseries querying
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timeseries_session ON sensor_timeseries(session_id, timestamp_ms)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_session ON physiological_metrics(session_id, timestamp_s)"
            )

            conn.commit()
            conn.close()

    # Participant Management
    def add_participant(
        self, name: str, embedding: Optional[np.ndarray] = None, notes: str = ""
    ) -> int:
        """Enroll or update a participant record with face embedding."""
        emb_json = json.dumps(embedding.tolist()) if embedding is not None else None
        now_str = datetime.now().isoformat()
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO participants (name, enrolled_at, embedding_json, notes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    enrolled_at = excluded.enrolled_at,
                    embedding_json = coalesce(excluded.embedding_json, participants.embedding_json),
                    notes = excluded.notes
                """,
                (name, now_str, emb_json, notes),
            )
            participant_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return participant_id

    def get_participant_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM participants WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        if res.get("embedding_json"):
            res["embedding"] = np.array(json.loads(res["embedding_json"]))
        return res

    def get_participant_by_id(self, participant_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM participants WHERE participant_id = ?", (participant_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        if res.get("embedding_json"):
            res["embedding"] = np.array(json.loads(res["embedding_json"]))
        return res

    def seed_default_participants(self) -> List[str]:
        """Ensures default authorized participants (Ashutosh and Dhrunil) exist with embeddings."""
        seeded = []
        # Check datasets/face_features/feature.npz
        feat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "face_features", "feature.npz")
        embs_cache = {}
        if os.path.exists(feat_path):
            try:
                data = np.load(feat_path, allow_pickle=True)
                names = data.get("images_name", [])
                embs = data.get("images_emb", [])
                for target_name in ["ashutosh", "dhrunil"]:
                    mask = np.array([str(n).strip().lower() == target_name.lower() for n in names])
                    if np.any(mask):
                        mean_emb = np.mean(embs[mask], axis=0)
                        norm = np.linalg.norm(mean_emb)
                        if norm > 0:
                            mean_emb /= norm
                        embs_cache[target_name] = mean_emb
            except Exception:
                pass

        for name, role in [("Ashutosh", "Lead Biomedical Research Engineer"), ("Dhrunil", "Lead Systems & Telemetry Architect")]:
            existing = self.get_participant_by_name(name)
            emb = embs_cache.get(name.lower())
            if emb is None:
                # Deterministic normalized unit vector fallback
                rng = np.random.RandomState(hash(name) % (2**32))
                emb = rng.randn(512).astype(np.float32)
                emb /= np.linalg.norm(emb)

            if not existing or existing.get("embedding") is None:
                self.add_participant(name, emb, notes=f"Enrolled Participant - {role}")
                seeded.append(name)
            elif emb is not None:
                self.add_participant(name, emb, notes=existing.get("notes") or f"Enrolled Participant - {role}")
                seeded.append(name)
        return seeded

    def get_all_participants(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM participants ORDER BY name ASC")
        rows = cursor.fetchall()
        conn.close()
        participants = []
        for r in rows:
            d = dict(r)
            if d.get("embedding_json"):
                d["embedding"] = np.array(json.loads(d["embedding_json"]))
            participants.append(d)
        return participants

    def delete_participant(self, participant_id: int) -> bool:
        """Delete participant and anonymize associated session logs."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM participants WHERE participant_id = ?",
                (participant_id,),
            )
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return deleted

    # Session Management
    def create_session(
        self, session_id: str, participant_id: Optional[int] = None, is_guest: bool = False
    ) -> str:
        now_str = datetime.now().isoformat()
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions (session_id, participant_id, is_guest, start_time, status)
                VALUES (?, ?, ?, ?, 'ACTIVE')
                """,
                (session_id, participant_id, 1 if is_guest else 0, now_str),
            )
            conn.commit()
            conn.close()
            return session_id

    def end_session(self, session_id: str, summary_data: Optional[Dict[str, Any]] = None):
        now = datetime.now()
        now_str = now.isoformat()
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            duration = 0.0
            if row and row["start_time"]:
                try:
                    start_dt = datetime.fromisoformat(row["start_time"])
                    duration = (now - start_dt).total_seconds()
                except Exception:
                    pass

            summary_str = json.dumps(summary_data) if summary_data else None
            cursor.execute(
                """
                UPDATE sessions
                SET end_time = ?, duration_seconds = ?, status = 'COMPLETED', summary_json = ?
                WHERE session_id = ?
                """,
                (now_str, duration, summary_str, session_id),
            )
            conn.commit()
            conn.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.*, p.name as participant_name
            FROM sessions s
            LEFT JOIN participants p ON s.participant_id = p.participant_id
            WHERE s.session_id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        if res.get("summary_json"):
            res["summary"] = json.loads(res["summary_json"])
        return res

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.*, p.name as participant_name
            FROM sessions s
            LEFT JOIN participants p ON s.participant_id = p.participant_id
            ORDER BY s.start_time DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("summary_json"):
                try:
                    d["summary"] = json.loads(d["summary_json"])
                except Exception:
                    d["summary"] = {}
            results.append(d)
        return results

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return deleted

    def purge_all_sessions(self) -> Dict[str, int]:
        """Purges all sessions, sensor timeseries, and metrics while preserving canonical enrolled participants."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sensor_timeseries")
            ts_deleted = cursor.rowcount
            cursor.execute("DELETE FROM physiological_metrics")
            pm_deleted = cursor.rowcount
            cursor.execute("DELETE FROM sessions")
            sess_deleted = cursor.rowcount
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('sensor_timeseries', 'physiological_metrics', 'sessions')")
            conn.commit()
            cursor.execute("VACUUM")
            conn.close()
            return {
                "sessions_deleted": sess_deleted,
                "timeseries_deleted": ts_deleted,
                "metrics_deleted": pm_deleted,
            }

    # Timeseries & Metrics Persistence
    def insert_sensor_samples(self, samples: List[Tuple[str, int, int, int, int, int, int]]):
        """Batch insert raw sensor telemetry."""
        if not samples:
            return
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO sensor_timeseries (session_id, timestamp_ms, seq, ecg_raw, ppg_red, ppg_ir, leads_off)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                samples,
            )
            conn.commit()
            conn.close()

    def insert_metric(self, session_id: str, metric: Dict[str, Any]):
        """Insert calculated biomedical metrics window."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO physiological_metrics (
                    session_id, timestamp_s, ecg_hr, ppg_hr, spo2, rr_interval_ms,
                    rmssd_ms, sdnn_ms, pat_ms, resp_rate_bpm, signal_quality, bp_systolic, bp_diastolic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    metric.get("timestamp_s", 0.0),
                    metric.get("ecg_hr"),
                    metric.get("ppg_hr"),
                    metric.get("spo2"),
                    metric.get("rr_interval_ms"),
                    metric.get("rmssd_ms"),
                    metric.get("sdnn_ms"),
                    metric.get("pat_ms"),
                    metric.get("resp_rate_bpm"),
                    metric.get("signal_quality", "N/A"),
                    metric.get("bp_systolic"),
                    metric.get("bp_diastolic"),
                ),
            )
            conn.commit()
            conn.close()

    def get_session_timeseries(self, session_id: str, limit: int = 10000) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ms, seq, ecg_raw, ppg_red, ppg_ir, leads_off
            FROM sensor_timeseries
            WHERE session_id = ?
            ORDER BY timestamp_ms ASC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_session_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM physiological_metrics
            WHERE session_id = ?
            ORDER BY timestamp_s ASC
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # Data Export Capabilities
    def export_session_timeseries_csv(self, session_id: str, filepath: Optional[str] = None) -> str:
        """Export session raw telemetry data to CSV string or write to file."""
        rows = self.get_session_timeseries(session_id, limit=1000000)
        output = io.StringIO()
        fieldnames = ["timestamp_ms", "seq", "ecg_raw", "ppg_red", "ppg_ir", "leads_off"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
        csv_str = output.getvalue()
        if filepath:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                f.write(csv_str)
        return csv_str

    def export_session_metrics_csv(self, session_id: str, filepath: Optional[str] = None) -> str:
        """Export session physiological metrics to CSV string or write to file."""
        rows = self.get_session_metrics(session_id)
        output = io.StringIO()
        fieldnames = [
            "id", "session_id", "timestamp_s", "ecg_hr", "ppg_hr", "spo2",
            "rr_interval_ms", "rmssd_ms", "sdnn_ms", "pat_ms", "resp_rate_bpm",
            "signal_quality", "bp_systolic", "bp_diastolic"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
        csv_str = output.getvalue()
        if filepath:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                f.write(csv_str)
        return csv_str

