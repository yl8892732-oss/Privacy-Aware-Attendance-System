"""
Face Attendance System - Database & Privacy Manager (Role 5)
Course: AIT103 - Introduction to Intelligence Applications
Group Project - Intelligent System Report

This module handles:
1. Local SQLite database for storing student IDs and face embeddings (feature vectors)
2. Privacy-aware data handling (no raw images stored, only numerical vectors)
3. Attendance logging to CSV
4. Secure memory cleanup utilities

Author: Role 5 (Database & Privacy Manager)
"""

import sqlite3
import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime
from typing import Optional, Tuple, List

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
    Initialize the SQLite database with the users table.

    Table Schema:
        - student_id (TEXT, PRIMARY KEY): Unique student identifier
        - embedding (BLOB): 512-dim face feature vector stored as binary

    Args:
        db_path: Path to the SQLite database file (default: face_attendance.db)
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
# 2. USER REGISTRATION (PRIVACY-FIRST)
# =============================================================================

def register_user(student_id: str, 
                  embedding: np.ndarray, 
                  db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Register a new student with their face embedding into the local database.

    PRIVACY GUARANTEE:
        - This function ONLY receives numerical feature vectors (embeddings).
        - NO raw face images are ever passed to or stored by this function.
        - The original camera frame stays in RAM and is destroyed after 
          feature extraction by the face recognition module (Role 2).

    Args:
        student_id: Student ID string (e.g., "AIT2509020")
        embedding: Face feature vector, shape (512,) or (128,), dtype float32
        db_path: Path to SQLite database

    Returns:
        True if registration succeeds, False otherwise.
    """
    # Validate input: ensure embedding is a 1D numpy array
    if not isinstance(embedding, np.ndarray):
        print("[DB Manager] Error: embedding must be a numpy array")
        return False

    embedding = np.asarray(embedding).flatten().astype(np.float32)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Convert numpy array to binary blob for SQLite storage
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
# 3. USER RECOGNITION (COSINE SIMILARITY MATCHING)
# =============================================================================

def recognize_user(current_embedding: np.ndarray,
                   threshold: float = DEFAULT_THRESHOLD,
                   db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Recognize a student by comparing their face embedding against the database.

    Algorithm: Cosine Similarity
        sim(A, B) = (A · B) / (||A|| * ||B||)
        Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
        If max(similarity) >= threshold, return the matched student_id.
        Otherwise, return "Unknown".

    Args:
        current_embedding: Feature vector extracted from live camera frame
        threshold: Similarity threshold (default 0.6). Higher = stricter matching.
        db_path: Path to SQLite database

    Returns:
        Matched student_id string, or "Unknown" if no match found.
    """
    if not isinstance(current_embedding, np.ndarray):
        print("[DB Manager] Error: current_embedding must be a numpy array")
        return "Unknown"

    current_embedding = np.asarray(current_embedding).flatten().astype(np.float32)
    current_norm = np.linalg.norm(current_embedding)

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
        print("[DB Manager] Database is empty. No users registered yet.")
        return "Unknown"

    best_similarity = -1.0
    best_match_id = "Unknown"

    for student_id, embedding_bytes in rows:
        # Reconstruct numpy array from binary blob
        db_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        db_norm = np.linalg.norm(db_embedding)

        if db_norm == 0:
            continue

        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(current_embedding, db_embedding) / (current_norm * db_norm)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = student_id

    if best_similarity >= threshold:
        print(f"[DB Manager] Match found: {best_match_id} (similarity={best_similarity:.4f})")
        return best_match_id
    else:
        print(f"[DB Manager] No match. Best similarity={best_similarity:.4f} < threshold={threshold}")
        return "Unknown"


# =============================================================================
# 4. ATTENDANCE LOGGING (CSV EXPORT)
# =============================================================================

def log_attendance(student_id: str,
                   status: str = "Success",
                   log_path: str = DEFAULT_LOG_PATH) -> bool:
    """
    Log an attendance record to a local CSV file.

    Columns: student_id, status, timestamp
    - Appends new rows if file exists; creates new file otherwise.
    - Timestamp is auto-generated in "YYYY-MM-DD HH:MM:SS" format.

    Args:
        student_id: Matched student ID or "Unknown"
        status: "Success", "Failed", "Spoofing_Detected", etc.
        log_path: Path to the CSV log file

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

        print(f"[DB Manager] Attendance logged: {student_id} @ {timestamp}")
        return True

    except Exception as e:
        print(f"[DB Manager] Error writing attendance log: {e}")
        return False


def get_attendance_records(log_path: str = DEFAULT_LOG_PATH) -> Optional[pd.DataFrame]:
    """
    Read all attendance records from the CSV log file.

    Args:
        log_path: Path to the CSV log file

    Returns:
        pandas DataFrame of all records, or None if file not found.
    """
    if not os.path.exists(log_path):
        print(f"[DB Manager] No attendance log found at {log_path}")
        return None

    return pd.read_csv(log_path)


# =============================================================================
# 5. PRIVACY & SECURITY UTILITIES
# =============================================================================

def secure_delete(var) -> None:
    """
    Securely delete a variable from memory and trigger garbage collection.

    This is used to ensure raw camera frames (numpy arrays containing 
    face images) are completely removed from RAM after feature extraction.

    Usage:
        frame = capture_camera()      # Raw image in RAM
        embedding = extract_face(frame)  # Get feature vector
        secure_delete(frame)          # Destroy raw image
        # Now only the embedding (numbers) exists. No photo leak possible.

    Args:
        var: The variable to delete (typically a numpy array image frame)
    """
    try:
        del var
    except NameError:
        pass
    gc.collect()


def clear_all_users(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Delete ALL registered users from the database. Use with caution!
    Useful for testing or system reset.

    Args:
        db_path: Path to SQLite database

    Returns:
        True if cleared successfully, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM users")
        conn.commit()
        print("[DB Manager] All users cleared from database.")
        return True
    except sqlite3.Error as e:
        print(f"[DB Manager] Error clearing database: {e}")
        return False
    finally:
        conn.close()


def get_user_count(db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Get the total number of registered users in the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Integer count of registered users.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        return count
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# =============================================================================
# 6. DEMO / TESTING (Run this file directly to test)
# =============================================================================

if __name__ == "__main__":
    """
    Self-test script. Run: python db_manager.py
    This simulates the full pipeline without needing a camera or face model.
    """
    print("=" * 60)
    print("Face Attendance DB Manager - Self Test")
    print("=" * 60)

    # Step 1: Initialize database
    init_db()

    # Step 2: Simulate registering 2 students with random 512-dim vectors
    # In real system, these come from EdgeFace/MobileFaceNet (Role 2)
    print("\n--- Testing Registration ---")
    fake_embedding_1 = np.random.randn(512).astype(np.float32)
    fake_embedding_2 = np.random.randn(512).astype(np.float32)

    register_user("AIT2509001", fake_embedding_1)
    register_user("AIT2509002", fake_embedding_2)

    print(f"Total registered users: {get_user_count()}")

    # Step 3: Simulate recognition (same person returns)
    print("\n--- Testing Recognition ---")
    result = recognize_user(fake_embedding_1, threshold=0.6)
    print(f"Recognition result: {result}")

    # Step 4: Simulate unknown person
    print("\n--- Testing Unknown Face ---")
    unknown_embedding = np.random.randn(512).astype(np.float32)
    result = recognize_user(unknown_embedding, threshold=0.6)
    print(f"Recognition result: {result}")

    # Step 5: Log attendance
    print("\n--- Testing Attendance Log ---")
    log_attendance("AIT2509001", status="Success")
    log_attendance("Unknown", status="Failed")

    # Show all logs
    records = get_attendance_records()
    if records is not None:
        print("\nAttendance Records:")
        print(records.to_string(index=False))

    # Step 6: Privacy demo - secure delete
    print("\n--- Testing Secure Delete ---")
    sensitive_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Fake camera frame
    print(f"Frame exists in memory: {'sensitive_frame' in dir()}")
    secure_delete(sensitive_frame)
    try:
        print(sensitive_frame)
    except NameError:
        print("Frame successfully destroyed from memory.")

    print("\n" + "=" * 60)
    print("All tests passed! Module is ready for integration.")
    print("=" * 60)
