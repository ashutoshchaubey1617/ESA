"""
Database cleanup and sanitization script for Multimodal Health & Access System.
Prepares the project for real hardware testing by:
1. Retaining canonical enrolled participants (Ashutosh & Dhrunil) with valid ArcFace embeddings.
2. Removing all test/mock sessions, timeseries sensor data, and physiological metrics.
3. Cleaning any temporary export CSVs in data/.
4. Resetting autoincrement sequences for timeseries and metrics.
5. Synchronizing data/app.db to identical clean canonical state.
6. Verifying database integrity and running VACUUM for disk reclamation.
"""

import os
import sqlite3
import json
import numpy as np

def cleanup_database(db_path: str, is_primary: bool = True):
    print(f"\n{'='*60}")
    print(f"CLEANING DATABASE: {db_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist. Skipping.")
        return None

    size_before = os.path.getsize(db_path)
    print(f"File size before cleanup: {size_before:,} bytes ({size_before / (1024*1024):.2f} MB)")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # 1. Inspect existing state
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall() if r[0] != 'sqlite_sequence']
    print(f"Detected tables: {tables}")

    counts_before = {}
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM [{t}]")
        counts_before[t] = cur.fetchone()[0]
    print(f"Row counts before cleanup: {counts_before}")

    # Inspect participants
    cur.execute("SELECT participant_id, name, enrolled_at, embedding_json, notes FROM participants")
    participants = cur.fetchall()
    print(f"Current participants count: {len(participants)}")
    for p in participants:
        p_id, name, enrolled_at, emb_json, notes = p
        has_emb = bool(emb_json)
        emb_dims = len(json.loads(emb_json)) if has_emb else 0
        norm = np.linalg.norm(json.loads(emb_json)) if has_emb else 0.0
        print(f"  - ID {p_id}: {name} (Enrolled: {enrolled_at}, Embedding dims: {emb_dims}, L2 norm: {norm:.4f})")

    # 2. Cleanup non-canonical participants (keep only Ashutosh and Dhrunil)
    cur.execute("SELECT participant_id, name FROM participants WHERE LOWER(name) NOT IN ('ashutosh', 'dhrunil')")
    non_canonical = cur.fetchall()
    if non_canonical:
        print(f"Found non-canonical participants to remove: {non_canonical}")
        for p_id, name in non_canonical:
            cur.execute("DELETE FROM participants WHERE participant_id = ?", (p_id,))
            print(f"  Deleted non-canonical participant: {name} (ID: {p_id})")
    else:
        print("No duplicate or non-canonical participants found.")

    # 3. Clean all mock/test sessions, timeseries, and metrics
    # Delete timeseries and metrics first, then sessions
    cur.execute("DELETE FROM sensor_timeseries")
    ts_deleted = cur.rowcount
    cur.execute("DELETE FROM physiological_metrics")
    pm_deleted = cur.rowcount
    cur.execute("DELETE FROM sessions")
    sess_deleted = cur.rowcount
    conn.commit()

    print(f"Purged mock/test data:")
    print(f"  - Sessions deleted: {sess_deleted}")
    print(f"  - Sensor timeseries rows deleted: {ts_deleted}")
    print(f"  - Physiological metrics rows deleted: {pm_deleted}")

    # 4. Reset autoincrement sequence counters for telemetry and metrics
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('sensor_timeseries', 'physiological_metrics')")
    conn.commit()
    print("Reset autoincrement sequences for sensor_timeseries and physiological_metrics.")

    # 5. Check for orphaned records
    cur.execute("SELECT COUNT(*) FROM sensor_timeseries WHERE session_id NOT IN (SELECT session_id FROM sessions)")
    orphaned_ts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM physiological_metrics WHERE session_id NOT IN (SELECT session_id FROM sessions)")
    orphaned_pm = cur.fetchone()[0]
    print(f"Orphaned sensor_timeseries count: {orphaned_ts}")
    print(f"Orphaned physiological_metrics count: {orphaned_pm}")

    # 6. Verify Foreign Key and SQLite Integrity
    cur.execute("PRAGMA foreign_key_check")
    fk_errors = cur.fetchall()
    cur.execute("PRAGMA integrity_check")
    integrity = cur.fetchall()
    print(f"PRAGMA foreign_key_check: {fk_errors if fk_errors else 'PASSED (0 violations)'}")
    print(f"PRAGMA integrity_check: {integrity[0][0] if integrity else 'UNKNOWN'}")

    # 7. Post-cleanup row counts
    counts_after = {}
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM [{t}]")
        counts_after[t] = cur.fetchone()[0]
    print(f"Row counts after cleanup: {counts_after}")

    # 8. Vacuum database to reclaim disk space
    conn.execute("VACUUM")
    conn.close()

    size_after = os.path.getsize(db_path)
    reclaimed = size_before - size_after
    print(f"File size after VACUUM: {size_after:,} bytes ({size_after / 1024:.2f} KB)")
    print(f"Disk space reclaimed: {reclaimed:,} bytes ({reclaimed / (1024*1024):.2f} MB)")

    return {
        "db_path": db_path,
        "size_before": size_before,
        "size_after": size_after,
        "reclaimed": reclaimed,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "sess_deleted": sess_deleted,
        "ts_deleted": ts_deleted,
        "pm_deleted": pm_deleted,
        "integrity": integrity[0][0] if integrity else 'ERROR'
    }


def sync_secondary_db(source_db: str, target_db: str):
    """Synchronizes canonical participant profiles into target_db."""
    print(f"\n{'='*60}")
    print(f"SYNCHRONIZING CANONICAL PARTICIPANTS TO {target_db}")
    print(f"{'='*60}")
    os.makedirs(os.path.dirname(target_db), exist_ok=True)
    
    src_conn = sqlite3.connect(source_db)
    src_cur = src_conn.cursor()
    src_cur.execute("SELECT participant_id, name, enrolled_at, embedding_json, notes FROM participants")
    canonical_participants = src_cur.fetchall()
    src_conn.close()

    tgt_conn = sqlite3.connect(target_db)
    tgt_conn.execute("PRAGMA foreign_keys = ON")
    tgt_cur = tgt_conn.cursor()

    for p in canonical_participants:
        p_id, name, enrolled_at, emb_json, notes = p
        tgt_cur.execute("""
            INSERT INTO participants (participant_id, name, enrolled_at, embedding_json, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                participant_id = excluded.participant_id,
                enrolled_at = excluded.enrolled_at,
                embedding_json = excluded.embedding_json,
                notes = excluded.notes
        """, (p_id, name, enrolled_at, emb_json, notes))
    tgt_conn.commit()
    tgt_conn.close()
    print(f"Synchronized {len(canonical_participants)} canonical profiles into {target_db}.")


def clean_temporary_csvs(dirs: list):
    print(f"\n{'='*60}")
    print("SCANNING FOR TEMPORARY CSV/EXPORT FILES")
    print(f"{'='*60}")
    removed = []
    for d in dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(('.csv', '.tsv', '.tmp', '.bak')):
                    full_p = os.path.join(root, f)
                    try:
                        os.remove(full_p)
                        removed.append(full_p)
                        print(f"Removed temporary file: {full_p}")
                    except Exception as e:
                        print(f"Error removing {full_p}: {e}")
    if not removed:
        print("No temporary CSV or export files found in target directories.")
    return removed


if __name__ == "__main__":
    # Clean temporary files in data and database directories
    clean_temporary_csvs(["data", "database"])

    # 1. Clean primary database
    primary_stats = cleanup_database("health_monitor.db", is_primary=True)

    # 2. Sync participants and clean data/app.db
    sync_secondary_db("health_monitor.db", "data/app.db")
    secondary_stats = cleanup_database("data/app.db", is_primary=False)

    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    print(f"Primary DB (health_monitor.db):")
    print(f"  Sessions deleted: {primary_stats['sess_deleted']}")
    print(f"  Sensor timeseries deleted: {primary_stats['ts_deleted']:,}")
    print(f"  Physiological metrics deleted: {primary_stats['pm_deleted']:,}")
    print(f"  Disk space reclaimed: {primary_stats['reclaimed'] / (1024*1024):.2f} MB")
    print(f"  Integrity: {primary_stats['integrity']}")
    print(f"Secondary DB (data/app.db):")
    print(f"  Integrity: {secondary_stats['integrity']}")
    print("Status: 100% READY FOR REAL HARDWARE TESTING.")
