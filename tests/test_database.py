"""
Unit tests for SQLite Database and Persistence Layer
"""

import os
import tempfile
import numpy as np
import pytest
from database.db import DatabaseManager


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


def test_participant_crud(temp_db):
    dummy_emb = np.random.randn(512).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)

    # 1. Add participant
    p_id = temp_db.add_participant(name="Alice Smith", embedding=dummy_emb, notes="Clinical Trial Subject")
    assert p_id is not None

    # 2. Retrieve participant
    p = temp_db.get_participant_by_name("Alice Smith")
    assert p is not None
    assert p["name"] == "Alice Smith"
    assert p["embedding"] is not None
    assert len(p["embedding"]) == 512
    assert np.allclose(p["embedding"], dummy_emb, atol=1e-5)

    # 3. Retrieve all
    all_p = temp_db.get_all_participants()
    assert len(all_p) == 1

    # 4. Delete participant
    deleted = temp_db.delete_participant(p_id)
    assert deleted is True
    assert temp_db.get_participant_by_name("Alice Smith") is None


def test_session_lifecycle_and_timeseries(temp_db):
    p_id = temp_db.add_participant(name="Bob Jones")

    # Start session
    session_id = "SESS_TEST_001"
    temp_db.create_session(session_id, participant_id=p_id, is_guest=False)

    # Insert raw telemetry
    samples = [
        (session_id, 1000 + i * 10, i, 512 + i, 60000 + i, 65000 + i, 0)
        for i in range(10)
    ]
    temp_db.insert_sensor_samples(samples)

    # Retrieve timeseries
    retrieved = temp_db.get_session_timeseries(session_id)
    assert len(retrieved) == 10
    assert retrieved[0]["ecg_raw"] == 512

    # Insert calculated metrics
    metric = {
        "timestamp_s": 12.5,
        "ecg_hr": 72.0,
        "ppg_hr": 73.0,
        "spo2": 98.0,
        "rmssd_ms": 35.0,
        "sdnn_ms": 42.0,
        "signal_quality": "EXCELLENT",
    }
    temp_db.insert_metric(session_id, metric)

    # End session
    temp_db.end_session(session_id, summary_data={"avg_hr": 72.5})
    sess = temp_db.get_session(session_id)
    assert sess["status"] == "COMPLETED"
    assert sess["summary"]["avg_hr"] == 72.5


def test_schemas_and_indexes(temp_db):
    conn = temp_db._get_connection()
    cursor = conn.cursor()

    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r["name"] for r in cursor.fetchall()}
    assert "participants" in tables
    assert "sessions" in tables
    assert "sensor_timeseries" in tables
    assert "physiological_metrics" in tables

    # Verify indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {r["name"] for r in cursor.fetchall()}
    assert "idx_timeseries_session" in indexes
    assert "idx_metrics_session" in indexes

    conn.close()


def test_foreign_key_cascade_and_set_null(temp_db):
    # 1. Test ON DELETE SET NULL on sessions when participant is deleted
    p_id = temp_db.add_participant("Charlie Brown")
    s_id = "SESS_CHARLIE_01"
    temp_db.create_session(s_id, participant_id=p_id)

    sess_before = temp_db.get_session(s_id)
    assert sess_before["participant_id"] == p_id
    assert sess_before["participant_name"] == "Charlie Brown"

    # Delete participant (privacy / right-to-be-forgotten)
    temp_db.delete_participant(p_id)

    # Session record still exists, but participant_id is SET NULL (anonymized)
    sess_after = temp_db.get_session(s_id)
    assert sess_after is not None
    assert sess_after["participant_id"] is None
    assert sess_after["participant_name"] is None

    # 2. Test ON DELETE CASCADE on timeseries and metrics when session is deleted
    samples = [(s_id, 1000 + i * 10, i, 500 + i, 60000, 65000, 0) for i in range(5)]
    temp_db.insert_sensor_samples(samples)
    temp_db.insert_metric(s_id, {"timestamp_s": 1.0, "ecg_hr": 75.0, "spo2": 99.0})

    assert len(temp_db.get_session_timeseries(s_id)) == 5
    assert len(temp_db.get_session_metrics(s_id)) == 1

    # Delete session
    temp_db.delete_session(s_id)

    # Verify cascading delete cleaned up all associated sensor rows and metrics
    assert len(temp_db.get_session_timeseries(s_id)) == 0
    assert len(temp_db.get_session_metrics(s_id)) == 0


def test_csv_export_capabilities(temp_db, tmp_path):
    s_id = "SESS_EXPORT_01"
    temp_db.create_session(s_id)

    samples = [(s_id, 1000 + i * 10, i, 512 + i, 61000, 62000, 0) for i in range(3)]
    temp_db.insert_sensor_samples(samples)

    metric = {
        "timestamp_s": 5.0,
        "ecg_hr": 70.0,
        "ppg_hr": 71.0,
        "spo2": 98.5,
        "rr_interval_ms": 857.0,
        "rmssd_ms": 32.0,
        "sdnn_ms": 40.0,
        "pat_ms": 230.0,
        "resp_rate_bpm": 15.0,
        "signal_quality": "EXCELLENT",
        "bp_systolic": 118.0,
        "bp_diastolic": 78.0,
    }
    temp_db.insert_metric(s_id, metric)

    # Test timeseries CSV export to string and file
    csv_ts_file = tmp_path / "ts_export.csv"
    ts_csv = temp_db.export_session_timeseries_csv(s_id, filepath=str(csv_ts_file))
    assert "timestamp_ms,seq,ecg_raw,ppg_red,ppg_ir,leads_off" in ts_csv
    assert "1000,0,512,61000,62000,0" in ts_csv
    assert csv_ts_file.exists()
    assert csv_ts_file.read_text(encoding="utf-8") == ts_csv

    # Test metrics CSV export to string and file
    csv_m_file = tmp_path / "metrics_export.csv"
    m_csv = temp_db.export_session_metrics_csv(s_id, filepath=str(csv_m_file))
    assert "ecg_hr" in m_csv
    assert "70.0" in m_csv
    assert "118.0" in m_csv
    assert csv_m_file.exists()
    assert csv_m_file.read_text(encoding="utf-8") == m_csv


def test_thread_safe_concurrent_writes(temp_db):
    import threading

    session_id = "SESS_THREAD_TEST"
    temp_db.create_session(session_id)

    errors = []

    def worker_insert(worker_idx):
        try:
            samples = [
                (session_id, 10000 + worker_idx * 100 + i * 10, i, 500, 60000, 60000, 0)
                for i in range(10)
            ]
            temp_db.insert_sensor_samples(samples)
            temp_db.insert_metric(
                session_id,
                {"timestamp_s": worker_idx + 0.1, "ecg_hr": 70.0 + worker_idx, "spo2": 99.0},
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker_insert, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    # 10 workers * 10 samples = 100 samples
    timeseries = temp_db.get_session_timeseries(session_id, limit=200)
    assert len(timeseries) == 100
    metrics = temp_db.get_session_metrics(session_id)
    assert len(metrics) == 10

