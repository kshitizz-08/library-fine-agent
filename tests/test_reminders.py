"""
Unit Tests for Due Date & Pre-Due Fine Reminder Feature
"""

import os
import shutil
import tempfile
import unittest
from datetime import date, timedelta

from reminders.reminder_engine import DueDateReminderManager, LOAN_PERIODS, DAILY_FINE_RATES
from agents.supervisor import LibrarySupervisor


class TestDueDateReminders(unittest.TestCase):
    """Test suite for Due Date and Pre-Due Fine Reminder logic."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_loans.db")
        self.manager = DueDateReminderManager(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_due_date_calculation_by_book_category(self):
        """Verify due date is calculated strictly per library borrowing policy."""
        today = date.today()

        # Novel: 14 days
        novel_loan = self.manager.issue_loan(
            member_id="MEM-TEST",
            book_title="Test Novel",
            book_type="novel",
            borrow_date_str=today.strftime("%Y-%m-%d"),
        )
        expected_novel_due = (today + timedelta(days=14)).strftime("%Y-%m-%d")
        self.assertEqual(novel_loan["due_date"], expected_novel_due)
        self.assertEqual(novel_loan["daily_rate"], 5)

        # Textbook: 7 days
        textbook_loan = self.manager.issue_loan(
            member_id="MEM-TEST",
            book_title="Test Textbook",
            book_type="textbook",
            borrow_date_str=today.strftime("%Y-%m-%d"),
        )
        expected_tb_due = (today + timedelta(days=7)).strftime("%Y-%m-%d")
        self.assertEqual(textbook_loan["due_date"], expected_tb_due)
        self.assertEqual(textbook_loan["daily_rate"], 10)

        # Reference: 2 days
        ref_loan = self.manager.issue_loan(
            member_id="MEM-TEST",
            book_title="Test Reference",
            book_type="reference",
            borrow_date_str=today.strftime("%Y-%m-%d"),
        )
        expected_ref_due = (today + timedelta(days=2)).strftime("%Y-%m-%d")
        self.assertEqual(ref_loan["due_date"], expected_ref_due)
        self.assertEqual(ref_loan["daily_rate"], 15)

    def test_pre_due_reminder_triggers(self):
        """Verify pre-due reminder is triggered when loan is within 2 days of due date."""
        today = date.today()

        # Loan due in 2 days (Borrowed 5 days ago for 7-day textbook)
        b_date = today - timedelta(days=5)
        loan = self.manager.issue_loan(
            member_id="MEM-PREDUE",
            book_title="Operating Systems",
            book_type="textbook",
            borrow_date_str=b_date.strftime("%Y-%m-%d"),
        )

        status = self.manager.evaluate_loan_status(loan, current_date=today)
        self.assertEqual(status["urgency"], "DUE_SOON")
        self.assertEqual(status["days_remaining"], 2)
        self.assertTrue(status["reminder_needed"])
        self.assertIn("late fine", status["alert_message"].lower())
        self.assertIn("10", status["alert_message"])

    def test_due_today_reminder(self):
        """Verify loan due today triggers urgent notification."""
        today = date.today()
        b_date = today - timedelta(days=7)  # Textbook borrowed 7 days ago
        loan = self.manager.issue_loan(
            member_id="MEM-TODAY",
            book_title="Database Systems",
            book_type="textbook",
            borrow_date_str=b_date.strftime("%Y-%m-%d"),
        )

        status = self.manager.evaluate_loan_status(loan, current_date=today)
        self.assertEqual(status["urgency"], "DUE_TODAY")
        self.assertEqual(status["days_remaining"], 0)
        self.assertTrue(status["reminder_needed"])
        self.assertIn("due today", status["alert_message"].lower())

    def test_overdue_alert_and_fine_accumulation(self):
        """Verify overdue loan computes fine liability correctly."""
        today = date.today()
        # Reference book borrowed 5 days ago (2-day loan period -> 3 days overdue)
        b_date = today - timedelta(days=5)
        loan = self.manager.issue_loan(
            member_id="MEM-LATE",
            book_title="Dictionary Vol 1",
            book_type="reference",
            borrow_date_str=b_date.strftime("%Y-%m-%d"),
        )

        status = self.manager.evaluate_loan_status(loan, current_date=today)
        self.assertEqual(status["urgency"], "OVERDUE")
        self.assertEqual(status["days_overdue"], 3)
        self.assertEqual(status["fine_accrued"], 3 * 15)  # 3 days * ₹15
        self.assertTrue(status["reminder_needed"])

    def test_send_reminder_notification(self):
        """Verify sending a reminder logs to notification audit table."""
        today = date.today()
        loan = self.manager.issue_loan(
            member_id="MEM-NOTIF",
            book_title="Sample Book",
            book_type="novel",
            borrow_date_str=today.strftime("%Y-%m-%d"),
        )

        res = self.manager.send_reminder(loan["loan_id"], channel="EMAIL")
        self.assertTrue(res["notification_id"].startswith("NOTIF-"))
        self.assertEqual(res["channel"], "EMAIL")

        logs = self.manager.list_notifications(member_id="MEM-NOTIF")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["loan_id"], loan["loan_id"])

    def test_return_loan_workflow(self):
        """Verify book return closes active loan and assesses any overdue fine."""
        today = date.today()
        loan = self.manager.issue_loan(
            member_id="MEM-RET",
            book_title="Python Deep Learning",
            book_type="textbook",
            borrow_date_str=(today - timedelta(days=10)).strftime("%Y-%m-%d"),
        )

        # Returned today (3 days overdue)
        ret = self.manager.return_loan(loan["loan_id"], return_date_str=today.strftime("%Y-%m-%d"))
        self.assertEqual(ret["status"], "RETURNED")
        self.assertEqual(ret["days_overdue"], 3)
        self.assertEqual(ret["fine_accrued"], 30)


if __name__ == "__main__":
    unittest.main()
