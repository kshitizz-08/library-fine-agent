"""
SQLite Memory & Member Profile Storage Module (Unit 3A & 3B)

- Unit 3A: LangGraph SQLite Checkpointing for Thread Memory
- Unit 3B: Composite Key (member_id, namespace) Member Profile Storage
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

from langgraph.checkpoint.sqlite import SqliteSaver

# Default storage paths
DEFAULT_PROFILE_DB = os.path.join(os.path.dirname(__file__), "..", "data", "member_profiles.db")
DEFAULT_CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints.db")


class MemberProfileStore:
    """
    Persistent member profile storage using SQLite.
    Composite primary key: (member_id, namespace) where namespace defaults to 'profile'.
    Uses parameterized SQL and JSON serialization.
    """

    def __init__(self, db_path: str = DEFAULT_PROFILE_DB):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the member_profiles table with composite primary key."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS member_profiles (
                    member_id TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'profile',
                    profile_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (member_id, namespace)
                )
                """
            )
            conn.commit()

    def get_profile(self, member_id: str, namespace: str = "profile") -> Dict[str, Any]:
        """
        Retrieve a member profile by (member_id, namespace).
        Returns an empty dict if no profile exists.
        """
        if not member_id:
            return {}

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT profile_json, updated_at
                FROM member_profiles
                WHERE member_id = ? AND namespace = ?
                """,
                (str(member_id).strip(), namespace),
            )
            row = cursor.fetchone()
            if row:
                try:
                    data = json.loads(row["profile_json"])
                    data["_updated_at"] = row["updated_at"]
                    return data
                except json.JSONDecodeError:
                    return {}
            return {}

    def save_profile(
        self,
        member_id: str,
        profile_data: Dict[str, Any],
        namespace: str = "profile",
    ) -> Dict[str, Any]:
        """
        Create or completely replace a member profile.
        """
        cleaned_id = str(member_id).strip()
        # Exclude internal metadata key if present
        data_to_save = {k: v for k, v in profile_data.items() if not k.startswith("_")}
        json_str = json.dumps(data_to_save, ensure_ascii=False)
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO member_profiles (member_id, namespace, profile_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(member_id, namespace) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (cleaned_id, namespace, json_str, now),
            )
            conn.commit()

        return self.get_profile(cleaned_id, namespace)

    def update_profile(
        self,
        member_id: str,
        partial_data: Dict[str, Any],
        namespace: str = "profile",
    ) -> Dict[str, Any]:
        """
        Update an existing profile without losing unrelated fields.
        Merges partial_data into current profile dictionary.
        """
        current_data = self.get_profile(member_id, namespace)
        # Strip internal keys
        merged_data = {k: v for k, v in current_data.items() if not k.startswith("_")}
        # Merge new partial data
        for k, v in partial_data.items():
            if not k.startswith("_"):
                merged_data[k] = v

        return self.save_profile(member_id, merged_data, namespace)

    def list_all_members(self, namespace: str = "profile") -> list:
        """List all member IDs in the specified namespace."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT member_id, updated_at FROM member_profiles WHERE namespace = ? ORDER BY updated_at DESC",
                (namespace,),
            )
            return [dict(row) for row in cursor.fetchall()]


# Module-level connection pool for SQLite checkpointing
_checkpoint_conn: Optional[sqlite3.Connection] = None
_checkpoint_saver: Optional[SqliteSaver] = None


def get_sqlite_checkpointer(db_path: str = DEFAULT_CHECKPOINT_DB) -> SqliteSaver:
    """
    Factory function for LangGraph SQLite Checkpointer.
    Ensures safe initialization and reuse of the SQLite connection.
    Separate from member-profile storage.
    """
    global _checkpoint_conn, _checkpoint_saver
    resolved_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    if _checkpoint_conn is None or _checkpoint_saver is None:
        _checkpoint_conn = sqlite3.connect(
            resolved_path,
            check_same_thread=False,
            autocommit=True
        )
        _checkpoint_saver = SqliteSaver(_checkpoint_conn)
        _checkpoint_saver.setup()

    return _checkpoint_saver


def close_checkpoint_connection() -> None:
    """Gracefully close the checkpointer SQLite connection."""
    global _checkpoint_conn, _checkpoint_saver
    if _checkpoint_conn is not None:
        try:
            _checkpoint_conn.close()
        except Exception:
            pass
        _checkpoint_conn = None
        _checkpoint_saver = None
