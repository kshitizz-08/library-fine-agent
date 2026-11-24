"""
Unit and Integration tests for LangGraph Supervisor Workflow (Unit 4 & Section 9).
Tests:
- Case 1: Normal fine calculation (Novel 5 days -> ₹25)
- Case 2: Thread memory follow-up (Reference book reuse -> ₹75)
- Case 3: Unknown policy rule rejection (POLICY_NOT_FOUND)
- Case 4: Fine waiver request & authorized staff approval
- Case 5: Negative days overdue error validation
- Additional: MAX_HOPS cutoff and unknown book types
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.supervisor import LibrarySupervisor
from approvals.waiver_approval import get_waiver_manager


class TestSupervisor(unittest.TestCase):
    """Evaluation test suite for College Review Cases 1 to 5."""

    def setUp(self):
        self.supervisor = LibrarySupervisor(max_hops=4)

    def test_case_1_normal_fine_calculation(self):
        """Case 1: Novel overdue by 5 days -> Policy verified rate ₹5 -> Fine = ₹25."""
        res = self.supervisor.invoke(
            "A novel was returned 5 days late. How much is the library fine?",
            member_id="MEM-CASE1",
            thread_id="thread-case-1",
        )
        self.assertEqual(res["policy_status"], "POLICY_FOUND")
        self.assertEqual(res["book_type"], "novel")
        self.assertEqual(res["overdue_days"], 5)
        self.assertIn("Fine: ₹25", res["final_response"])
        self.assertIn("data/library_rules.txt", res["final_response"])
        self.assertLessEqual(res["hop_count"], 4)

    def test_case_2_thread_memory_followup(self):
        """Case 2: Turn 1 novel 5 days -> Turn 2 reference follow-up reuses 5 days -> Fine = ₹75."""
        thread_id = "thread-case-2-mem"

        # Turn 1
        res1 = self.supervisor.invoke("My novel is 5 days overdue.", thread_id=thread_id)
        self.assertIn("Fine: ₹25", res1["final_response"])

        # Turn 2: Follow-up without repeating 5 days
        res2 = self.supervisor.invoke("And if it were a reference book?", thread_id=thread_id)
        self.assertIn("Fine: ₹75", res2["final_response"])
        self.assertEqual(res2["book_type"], "reference")
        self.assertEqual(res2["overdue_days"], 5)
        self.assertEqual(res2["policy_status"], "POLICY_FOUND")

    def test_case_3_unknown_policy_rule(self):
        """Case 3: Ask for free replacement book for damaged reference -> POLICY_NOT_FOUND."""
        res = self.supervisor.invoke(
            "Does the library provide free replacement books for damaged reference books?",
            thread_id="thread-case-3-unk",
        )
        self.assertEqual(res["policy_status"], "POLICY_NOT_FOUND")
        self.assertIn("POLICY_NOT_FOUND", res["final_response"])
        self.assertIn("does not provide free replacement books", res["final_response"].lower())

    def test_case_4_waiver_approval_workflow(self):
        """Case 4: Waiver request creates PENDING record; authorized staff can approve/reject."""
        waiver_mgr = get_waiver_manager()
        res = self.supervisor.invoke(
            "I need a fine waiver for my overdue textbook because I was in the hospital.",
            member_id="MEM-CASE4",
            thread_id="thread-case-4",
        )
        self.assertEqual(res["approval_status"], "PENDING")
        self.assertIn("PENDING STAFF REVIEW", res["final_response"])
        self.assertIn("WVR-", res["final_response"])

        # Extract request ID from trace
        trace = res["execution_trace"]
        waiver_step = [s for s in trace if s.get("agent") == "waiver_handler"][0]
        req_id = waiver_step["request_id"]

        # 1. Unauthorized reviewer must be rejected
        with self.assertRaises(PermissionError):
            waiver_mgr.decide_waiver(req_id, "staff_x", "APPROVED", "Approved", is_authorized=False)

        # 2. Authorized reviewer can approve
        decision_res = waiver_mgr.decide_waiver(
            req_id, "librarian_patel", "APPROVED", "Medical note verified", is_authorized=True
        )
        self.assertEqual(decision_res["status"], "APPROVED")
        self.assertEqual(decision_res["reviewer_id"], "librarian_patel")

        # 3. Duplicate decision must be rejected
        with self.assertRaises(ValueError):
            waiver_mgr.decide_waiver(req_id, "librarian_sharma", "REJECTED", "Override attempt", is_authorized=True)

    def test_case_5_invalid_input(self):
        """Case 5: Negative overdue days (-3) -> validation error."""
        res = self.supervisor.invoke(
            "Calculate the fine for a novel that is -3 days overdue.",
            thread_id="thread-case-5-err",
        )
        self.assertIn("days_overdue cannot be negative", res["final_response"])

    def test_max_hops_limit_cutoff(self):
        """Case: Verify workflow stops deterministically when MAX_HOPS is reached."""
        limited_supervisor = LibrarySupervisor(max_hops=1)
        res = limited_supervisor.invoke(
            "A novel was returned 5 days late.",
            thread_id="thread-hops-test",
            max_hops=1,
        )
        self.assertIn("MAX_HOPS", res["final_response"])

    def test_unknown_book_type(self):
        """Case: Unsupported book category returns validation error without hallucinating fine."""
        res = self.supervisor.invoke(
            "Calculate the fine for a magazine that is 4 days overdue.",
            thread_id="thread-unk-type",
        )
        self.assertIn("Unknown book_type", res["final_response"])


if __name__ == "__main__":
    unittest.main()
