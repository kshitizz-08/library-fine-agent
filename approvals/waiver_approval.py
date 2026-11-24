"""
Human-in-the-Loop Waiver Approval Module (Unit 4 / Section 7)

- Server-side enforcement of authorized staff review
- Rejection permanence (rejected stays rejected)
- Pending state enforcement (never represented as approved)
- Prevention of duplicate/conflicting decisions
- Full audit logging (reviewer, timestamp, reason)
"""

import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_WAIVER_DB = os.path.join(os.path.dirname(__file__), "..", "data", "waivers.db")


class WaiverManager:
    """
    Manages persistent fine waiver requests with human-in-the-loop staff approvals.
    """

    def __init__(self, db_path: str = DEFAULT_WAIVER_DB):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create waiver_requests table."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS waiver_requests (
                    request_id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    book_type TEXT NOT NULL,
                    days_overdue INTEGER NOT NULL,
                    fine_amount REAL NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    policy_evidence TEXT,
                    reviewer_id TEXT,
                    decision_reason TEXT,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    decided_at TIMESTAMP
                )
                """
            )
            conn.commit()

    def create_waiver_request(
        self,
        member_id: str,
        thread_id: str,
        book_type: str,
        days_overdue: int,
        fine_amount: float,
        reason: str,
        policy_evidence: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new waiver request in PENDING state.
        No waiver is granted automatically.
        """
        request_id = f"WVR-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO waiver_requests (
                    request_id, member_id, thread_id, book_type,
                    days_overdue, fine_amount, reason, status,
                    policy_evidence, requested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (
                    request_id,
                    str(member_id).strip(),
                    str(thread_id).strip(),
                    str(book_type).strip().lower(),
                    int(days_overdue),
                    float(fine_amount),
                    str(reason).strip(),
                    str(policy_evidence).strip(),
                    now,
                ),
            )
            conn.commit()

        return self.get_waiver_request(request_id)

    def get_waiver_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific waiver request by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM waiver_requests WHERE request_id = ?",
                (str(request_id).strip(),),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def list_waivers(
        self,
        status: Optional[str] = None,
        member_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List waivers with optional filters."""
        query = "SELECT * FROM waiver_requests WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status.upper())
        if member_id:
            query += " AND member_id = ?"
            params.append(str(member_id).strip())

        query += " ORDER BY requested_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def decide_waiver(
        self,
        request_id: str,
        reviewer_id: str,
        decision: str,
        decision_reason: str,
        is_authorized: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a human staff review decision.
        Enforces authorization, valid transitions, and non-duplicate decisions.
        """
        if not is_authorized:
            raise PermissionError("Unauthorized: Only authorized library staff can approve or reject fine waivers.")

        normalized_decision = decision.upper().strip()
        if normalized_decision not in ("APPROVED", "REJECTED"):
            raise ValueError(f"Invalid decision '{decision}'. Must be 'APPROVED' or 'REJECTED'.")

        current = self.get_waiver_request(request_id)
        if not current:
            raise KeyError(f"Waiver request '{request_id}' not found.")

        current_status = current["status"]
        if current_status != "PENDING":
            raise ValueError(
                f"Waiver request {request_id} has already been finalized as {current_status}. "
                "Decisions cannot be altered or duplicated."
            )

        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE waiver_requests
                SET status = ?,
                    reviewer_id = ?,
                    decision_reason = ?,
                    decided_at = ?
                WHERE request_id = ? AND status = 'PENDING'
                """,
                (
                    normalized_decision,
                    str(reviewer_id).strip(),
                    str(decision_reason).strip(),
                    now,
                    request_id,
                ),
            )
            conn.commit()

        return self.get_waiver_request(request_id)


_global_waiver_mgr: Optional[WaiverManager] = None


def get_waiver_manager() -> WaiverManager:
    """Singleton getter for WaiverManager."""
    global _global_waiver_mgr
    if _global_waiver_mgr is None:
        _global_waiver_mgr = WaiverManager()
    return _global_waiver_mgr
