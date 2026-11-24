"""
Due Date & Pre-Due Fine Reminder Engine
Units 3 & 4 Library Fine Calculator Agent Architecture

Features:
- Tracks active borrower loans with book classification (Novel, Textbook, Reference)
- Calculates due dates strictly aligned with data/library_rules.txt:
    * Novel: 14 days loan period, ₹5/day fine if overdue
    * Textbook: 7 days loan period, ₹10/day fine if overdue
    * Reference: 2 days loan period, ₹15/day fine if overdue
- Proactively generates pre-due reminders before the deadline (e.g. 2 days / 1 day / due today)
    warning borrowers about upcoming due dates and per-day fine consequences
- Detects overdue books and computes accumulating fine liability
- Dispatches and logs notifications (In-App, Simulated Email/SMS) into SQLite
"""

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loans.db")

LOAN_PERIODS = {
    "novel": 14,
    "textbook": 7,
    "reference": 2,
}

DAILY_FINE_RATES = {
    "novel": 5,
    "textbook": 10,
    "reference": 15,
}


class DueDateReminderManager:
    """Manages active loans, due date calculations, and pre-due fine reminders."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._seed_demo_loans_if_empty()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


    def _init_db(self):
        """Create tables for active loans and notification audit logs."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS loans (
                    loan_id TEXT PRIMARY KEY,
                    member_id TEXT NOT NULL,
                    book_title TEXT NOT NULL,
                    book_type TEXT NOT NULL,
                    borrow_date TEXT NOT NULL,
                    due_date TEXT NOT NULL,
                    daily_rate INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    returned_date TEXT,
                    reminder_sent_count INTEGER DEFAULT 0,
                    last_reminder_at TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminder_notifications (
                    notification_id TEXT PRIMARY KEY,
                    loan_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    reminder_type TEXT NOT NULL,
                    days_remaining INTEGER NOT NULL,
                    daily_rate INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    sent_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def issue_loan(
        self,
        member_id: str,
        book_title: str,
        book_type: str,
        borrow_date_str: Optional[str] = None,
        custom_loan_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Issue a new book loan.
        Calculates the due date deterministically based on book category:
        Novel (14d), Textbook (7d), Reference (2d).
        """
        b_type = book_type.lower().strip()
        if b_type not in LOAN_PERIODS:
            raise ValueError(f"Unsupported book category '{book_type}'. Must be one of: {list(LOAN_PERIODS.keys())}")

        loan_days = custom_loan_days if custom_loan_days is not None else LOAN_PERIODS[b_type]
        daily_rate = DAILY_FINE_RATES[b_type]

        if borrow_date_str:
            b_date = datetime.strptime(borrow_date_str, "%Y-%m-%d").date()
        else:
            b_date = date.today()

        due_date = b_date + timedelta(days=loan_days)
        loan_id = f"LOAN-{uuid.uuid4().hex[:8].upper()}"
        now_str = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO loans (
                    loan_id, member_id, book_title, book_type,
                    borrow_date, due_date, daily_rate, status,
                    reminder_sent_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', 0, ?)
                """,
                (
                    loan_id,
                    member_id.strip(),
                    book_title.strip(),
                    b_type,
                    b_date.strftime("%Y-%m-%d"),
                    due_date.strftime("%Y-%m-%d"),
                    daily_rate,
                    now_str,
                ),
            )
            conn.commit()

        return self.get_loan(loan_id)

    def get_loan(self, loan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a loan by loan_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM loans WHERE loan_id = ?", (loan_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def return_loan(self, loan_id: str, return_date_str: Optional[str] = None) -> Dict[str, Any]:
        """Mark a loan as returned and compute any applicable overdue fine."""
        loan = self.get_loan(loan_id)
        if not loan:
            raise ValueError(f"Loan ID '{loan_id}' not found.")
        if loan["status"] == "RETURNED":
            return loan

        ret_date = datetime.strptime(return_date_str, "%Y-%m-%d").date() if return_date_str else date.today()
        due_date = datetime.strptime(loan["due_date"], "%Y-%m-%d").date()
        days_overdue = max(0, (ret_date - due_date).days)
        fine_accrued = days_overdue * loan["daily_rate"]

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE loans
                SET status = 'RETURNED', returned_date = ?
                WHERE loan_id = ?
                """,
                (ret_date.strftime("%Y-%m-%d"), loan_id),
            )
            conn.commit()

        updated = self.get_loan(loan_id)
        updated["days_overdue"] = days_overdue
        updated["fine_accrued"] = fine_accrued
        return updated

    def evaluate_loan_status(self, loan: Dict[str, Any], current_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Evaluate loan status relative to today:
        - DUE_SOON (<= 2 days remaining) -> Pre-due reminder trigger
        - DUE_TODAY (0 days remaining) -> Urgent pre-due reminder trigger
        - OVERDUE (< 0 days remaining) -> Overdue penalty alert
        - ON_TRACK (> 2 days remaining) -> Healthy loan
        - RETURNED -> Closed loan
        """
        if loan["status"] == "RETURNED":
            return {
                **loan,
                "urgency": "RETURNED",
                "days_remaining": 0,
                "days_overdue": 0,
                "fine_accrued": 0,
                "reminder_needed": False,
                "alert_message": f"Book returned on {loan.get('returned_date', 'N/A')}.",
            }

        ref_date = current_date or date.today()
        due_date = datetime.strptime(loan["due_date"], "%Y-%m-%d").date()
        days_remaining = (due_date - ref_date).days
        daily_rate = loan["daily_rate"]
        book_title = loan["book_title"]
        book_type = loan["book_type"].capitalize()

        if days_remaining < 0:
            days_overdue = abs(days_remaining)
            fine_accrued = days_overdue * daily_rate
            urgency = "OVERDUE"
            reminder_needed = True
            alert_message = (
                f"[OVERDUE NOTICE] '{book_title}' ({book_type}) was due on {loan['due_date']} "
                f"({days_overdue} days overdue). Current fine is Rs. {fine_accrued} (accruing at Rs. {daily_rate}/day). "
                f"Please return immediately."
            )
        elif days_remaining == 0:
            urgency = "DUE_TODAY"
            reminder_needed = True
            fine_accrued = 0
            days_overdue = 0
            alert_message = (
                f"[URGENT DUE TODAY] '{book_title}' ({book_type}) is due TODAY ({loan['due_date']})! "
                f"Please return it before the desk closes, otherwise a fine of Rs. {daily_rate}/day will be charged."
            )
        elif days_remaining <= 2:
            urgency = "DUE_SOON"
            reminder_needed = True
            fine_accrued = 0
            days_overdue = 0
            alert_message = (
                f"[PRE-DUE REMINDER] '{book_title}' ({book_type}) is due in {days_remaining} day(s) "
                f"on {loan['due_date']}. Please submit on or before the due date, otherwise a late fine of "
                f"Rs. {daily_rate}/day will be charged."
            )
        else:
            urgency = "ON_TRACK"
            reminder_needed = False
            fine_accrued = 0
            days_overdue = 0
            alert_message = (
                f"[ACTIVE LOAN] '{book_title}' ({book_type}) is due in {days_remaining} days on {loan['due_date']}."
            )

        return {
            **loan,
            "urgency": urgency,
            "days_remaining": days_remaining,
            "days_overdue": days_overdue,
            "fine_accrued": fine_accrued,
            "reminder_needed": reminder_needed,
            "alert_message": alert_message,
        }

    def get_member_loans(self, member_id: str, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all loans for a member, enriched with dynamic reminder status."""
        query = "SELECT * FROM loans WHERE member_id = ?"
        params: List[Any] = [member_id.strip()]

        if status_filter:
            query += " AND status = ?"
            params.append(status_filter.upper())

        query += " ORDER BY due_date ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        return [self.evaluate_loan_status(dict(r)) for r in rows]

    def get_active_reminders_for_member(self, member_id: str) -> List[Dict[str, Any]]:
        """Retrieve only active loans requiring a pre-due or overdue reminder for a member."""
        loans = self.get_member_loans(member_id, status_filter="ACTIVE")
        return [l for l in loans if l["reminder_needed"]]

    def send_reminder(
        self,
        loan_id: str,
        channel: str = "EMAIL",
        custom_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record that a pre-due date reminder was delivered to the borrower
        (Simulates Email, SMS, or In-App Notification).
        """
        loan = self.get_loan(loan_id)
        if not loan:
            raise ValueError(f"Loan ID '{loan_id}' not found.")

        evaluated = self.evaluate_loan_status(loan)
        msg = custom_message or evaluated["alert_message"]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        notif_id = f"NOTIF-{uuid.uuid4().hex[:8].upper()}"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reminder_notifications (
                    notification_id, loan_id, member_id, reminder_type,
                    days_remaining, daily_rate, message, channel, sent_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notif_id,
                    loan_id,
                    loan["member_id"],
                    evaluated["urgency"],
                    evaluated["days_remaining"],
                    loan["daily_rate"],
                    msg,
                    channel.upper(),
                    now_str,
                ),
            )
            conn.execute(
                """
                UPDATE loans
                SET reminder_sent_count = reminder_sent_count + 1, last_reminder_at = ?
                WHERE loan_id = ?
                """,
                (now_str, loan_id),
            )
            conn.commit()

        return {
            "notification_id": notif_id,
            "loan_id": loan_id,
            "member_id": loan["member_id"],
            "urgency": evaluated["urgency"],
            "channel": channel.upper(),
            "message": msg,
            "sent_at": now_str,
        }

    def list_all_loans(self) -> List[Dict[str, Any]]:
        """List all loans in the database with evaluated status."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM loans ORDER BY due_date ASC").fetchall()
        return [self.evaluate_loan_status(dict(r)) for r in rows]

    def list_notifications(self, member_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List reminder notification history."""
        query = "SELECT * FROM reminder_notifications"
        params: List[Any] = []
        if member_id:
            query += " WHERE member_id = ?"
            params.append(member_id.strip())
        query += " ORDER BY sent_at DESC LIMIT 50"

        with self._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def _seed_demo_loans_if_empty(self):
        """Seed realistic loans for MEM-001 for live testing and review demonstration."""
        with self._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
            if count > 0:
                return

        today = date.today()

        # Loan 1: Textbook due in 2 days (Triggers PRE-DUE REMINDER)
        # Textbook loan period is 7 days; borrowed 5 days ago
        borrow_1 = today - timedelta(days=5)
        due_1 = borrow_1 + timedelta(days=LOAN_PERIODS["textbook"])

        # Loan 2: Novel due in 1 day (Triggers URGENT PRE-DUE REMINDER)
        # Novel loan period is 14 days; borrowed 13 days ago
        borrow_2 = today - timedelta(days=13)
        due_2 = borrow_2 + timedelta(days=LOAN_PERIODS["novel"])

        # Loan 3: Reference Book overdue by 2 days (Triggers OVERDUE FINE NOTICE)
        # Reference loan period is 2 days; borrowed 4 days ago
        borrow_3 = today - timedelta(days=4)
        due_3 = borrow_3 + timedelta(days=LOAN_PERIODS["reference"])

        # Loan 4: Novel on track (Due in 10 days, Healthy)
        borrow_4 = today - timedelta(days=4)
        due_4 = borrow_4 + timedelta(days=LOAN_PERIODS["novel"])

        demo_records = [
            (
                "LOAN-MEM1-001",
                "MEM-001",
                "Operating Systems: Three Easy Pieces",
                "textbook",
                borrow_1.strftime("%Y-%m-%d"),
                due_1.strftime("%Y-%m-%d"),
                DAILY_FINE_RATES["textbook"],
            ),
            (
                "LOAN-MEM1-002",
                "MEM-001",
                "The Great Gatsby",
                "novel",
                borrow_2.strftime("%Y-%m-%d"),
                due_2.strftime("%Y-%m-%d"),
                DAILY_FINE_RATES["novel"],
            ),
            (
                "LOAN-MEM1-003",
                "MEM-001",
                "Oxford English Reference Lexicon (Vol 3)",
                "reference",
                borrow_3.strftime("%Y-%m-%d"),
                due_3.strftime("%Y-%m-%d"),
                DAILY_FINE_RATES["reference"],
            ),
            (
                "LOAN-MEM1-004",
                "MEM-001",
                "To Kill a Mockingbird",
                "novel",
                borrow_4.strftime("%Y-%m-%d"),
                due_4.strftime("%Y-%m-%d"),
                DAILY_FINE_RATES["novel"],
            ),
        ]

        now_str = datetime.now().isoformat()
        with self._get_connection() as conn:
            for l_id, m_id, title, b_type, b_date, d_date, rate in demo_records:
                conn.execute(
                    """
                    INSERT INTO loans (
                        loan_id, member_id, book_title, book_type,
                        borrow_date, due_date, daily_rate, status,
                        reminder_sent_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', 0, ?)
                    """,
                    (l_id, m_id, title, b_type, b_date, d_date, rate, now_str),
                )
            conn.commit()


_REMINDER_MANAGER_INSTANCE: Optional[DueDateReminderManager] = None


def get_reminder_manager(db_path: str = DEFAULT_DB_PATH) -> DueDateReminderManager:
    """Singleton getter for DueDateReminderManager."""
    global _REMINDER_MANAGER_INSTANCE
    if _REMINDER_MANAGER_INSTANCE is None:
        _REMINDER_MANAGER_INSTANCE = DueDateReminderManager(db_path=db_path)
    return _REMINDER_MANAGER_INSTANCE
