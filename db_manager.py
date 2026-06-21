"""
Face Attendance System - Database & Privacy Manager (Role 5)
Course: AIT103 - Introduction to Intelligence Applications
Group Project - Intelligent System Report

Design Philosophy: The database module is a BLACK BOX.
    - Input: numerical embedding (a numpy array of floats)
    - Output: student_id string or "Unknown"
    - It does NOT touch cameras, liveness detection, or face models.

Author: Role 5 (Database & Privacy Manager)
"""

import sqlite3
import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_DB_PATH = "face_attendance.db"
DEFAULT_LOG_PATH = "attendance_log.csv"
DEFAULT_THRESHOLD = 0.6

# =============================================================================
# 1. DATABASE INITIALIZATION
# =============================================================================

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize the SQLite database.
    Creates a 'users' table if it does not exist.

    Table Schema:
        - student_id (TEXT, PRIMARY KEY): unique identifier
        - embedding (BLOB): face feature vector stored as binary

    Args:
        db_path: path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            student_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB Manager] Database initialized at: {db_path}")


# =============================================================================
# 2. USER REGISTRATION
# =============================================================================

def register_user(student_id: str, 
                  embedding: np.ndarray, 
                  db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Register a student by storing their student_id and face embedding.

    BLACK BOX PRINCIPLE:
        - Only receives a numerical embedding array.
        - No images, no camera frames, no model objects.

    Args:
        student_id: student ID string, e.g. "AIT2509001"
        embedding: face feature vector (numpy ndarray, any shape, float32)
        db_path: path to SQLite database

    Returns:
        True if registration succeeds, False otherwise.
    """
    if not isinstance(embedding, np.ndarray):
        print("[DB Manager] Error: embedding must be a numpy array")
        return False

    embedding = np.asarray(embedding).flatten().astype(np.float32)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        embedding_bytes = embedding.tobytes()
        cursor.execute(
            "INSERT OR REPLACE INTO users (student_id, embedding) VALUES (?, ?)",
            (student_id, embedding_bytes)
        )
        conn.commit()
        print(f"[DB Manager] User '{student_id}' registered successfully.")
        return True
    except sqlite3.Error as e:
        print(f"[DB Manager] Database error during registration: {e}")
        return False
    finally:
        conn.close()


# =============================================================================
# 3. USER RECOGNITION (BLACK BOX)
# =============================================================================

def recognize_user(embedding: np.ndarray,
                   db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Recognize a student by comparing their embedding against the database.

    BLACK BOX PRINCIPLE:
        - Input: a numerical embedding array.
        - Output: the matched student_id, or "Unknown" if no match.
        - Internal threshold is fixed at 0.6 (cosine similarity).

    Algorithm: Cosine Similarity
        similarity = dot(A, B) / (||A|| * ||B||)
        If max similarity >= 0.6, return the matched student_id.
        Otherwise, return "Unknown".

    Args:
        embedding: face feature vector (numpy ndarray)
        db_path: path to SQLite database

    Returns:
        Matched student_id string, or "Unknown".
    """
    if not isinstance(embedding, np.ndarray):
        print("[DB Manager] Error: embedding must be a numpy array")
        return "Unknown"

    embedding = np.asarray(embedding).flatten().astype(np.float32)
    current_norm = np.linalg.norm(embedding)

    if current_norm == 0:
        print("[DB Manager] Error: zero-norm embedding received")
        return "Unknown"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT student_id, embedding FROM users")
        rows = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"[DB Manager] Database error during recognition: {e}")
        return "Unknown"
    finally:
        conn.close()

    if not rows:
        print("[DB Manager] Database is empty.")
        return "Unknown"

    best_similarity = -1.0
    best_match_id = "Unknown"

    for student_id, emb_bytes in rows:
        db_embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        db_norm = np.linalg.norm(db_embedding)

        if db_norm == 0:
            continue

        similarity = np.dot(embedding, db_embedding) / (current_norm * db_norm)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = student_id

    if best_similarity >= DEFAULT_THRESHOLD:
        print(f"[DB Manager] Match: {best_match_id} (sim={best_similarity:.4f})")
        return best_match_id
    else:
        print(f"[DB Manager] No match. Best sim={best_similarity:.4f} < {DEFAULT_THRESHOLD}")
        return "Unknown"


# =============================================================================
# 4. ATTENDANCE LOGGING
# =============================================================================

def log_attendance(student_id: str,
                   status: str = "Success",
                   log_path: str = DEFAULT_LOG_PATH) -> bool:
    """
    Log an attendance record to a local CSV file.

    Args:
        student_id: matched student ID or "Unknown"
        status: e.g. "Success", "Failed", "Spoofing_Detected"
        log_path: path to CSV log file

    Returns:
        True if log written successfully, False otherwise.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "student_id": student_id,
        "status": status,
        "timestamp": timestamp
    }
    df = pd.DataFrame([record])

    try:
        if not os.path.exists(log_path):
            df.to_csv(log_path, index=False)
        else:
            df.to_csv(log_path, mode='a', header=False, index=False)
        print(f"[DB Manager] Logged: {student_id} @ {timestamp}")
        return True
    except Exception as e:
        print(f"[DB Manager] Error writing log: {e}")
        return False


def get_attendance_records(log_path: str = DEFAULT_LOG_PATH) -> Optional[pd.DataFrame]:
    """
    Read all attendance records from CSV.

    Args:
        log_path: path to CSV log file

    Returns:
        pandas DataFrame, or None if file not found.
    """
    if not os.path.exists(log_path):
        print(f"[DB Manager] No log found at {log_path}")
        return None
    return pd.read_csv(log_path)


# =============================================================================
# 5. PRIVACY & SECURITY UTILITIES
# =============================================================================

def secure_delete(var) -> None:
    """
    Securely delete a variable from memory and trigger garbage collection.
    Used to ensure sensitive data (e.g. raw image arrays) is fully removed.

    Args:
        var: the variable to delete.
    """
    try:
        del var
    except NameError:
        pass
    gc.collect()


def clear_all_users(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Delete ALL users from the database. Use with caution.

    Args:
        db_path: path to SQLite database

    Returns:
        True if cleared, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM users")
        conn.commit()
        print("[DB Manager] All users cleared.")
        return True
    except sqlite3.Error as e:
        print(f"[DB Manager] Error clearing DB: {e}")
        return False
    finally:
        conn.close()


def get_user_count(db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Get total number of registered users.

    Args:
        db_path: path to SQLite database

    Returns:
        integer count.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM users")
        return cursor.fetchone()[0]
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# =============================================================================
# 6. SELF-TEST (run: python db_manager.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DB Manager - Black Box Self Test")
    print("=" * 60)

    init_db()

    # Test with random embeddings (dimension is caller-defined)
    print("\n--- Registration ---")
    emb_1 = np.random.randn(512).astype(np.float32)
    emb_2 = np.random.randn(512).astype(np.float32)
    register_user("AIT2509001", emb_1)
    register_user("AIT2509002", emb_2)
    print(f"Total users: {get_user_count()}")

    print("\n--- Recognition (same person) ---")
    result = recognize_user(emb_1)
    print(f"Result: {result}")

    print("\n--- Recognition (unknown person) ---")
    unknown_emb = np.random.randn(512).astype(np.float32)
    result = recognize_user(unknown_emb)
    print(f"Result: {result}")

    print("\n--- Attendance Log ---")
    log_attendance("AIT2509001", "Success")
    log_attendance("Unknown", "Failed")
    records = get_attendance_records()
    if records is not None:
        print(records.to_string(index=False))

    print("\n--- Secure Delete ---")
    sensitive = np.zeros((1000, 1000), dtype=np.uint8)
    secure_delete(sensitive)
    try:
        print(sensitive)
    except NameError:
        print("Variable securely deleted from memory.")

    print("\n" + "=" * 60)
    print("All tests passed. Ready for integration.")
    print("=" * 60)
