"""
Unit tests for Local RAG Policy Retrieval (Unit 3C).
Tests policy indexing, novel/textbook/reference rates, and POLICY_NOT_FOUND rejection.
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag.rag_engine import get_policy_rag


class TestRAG(unittest.TestCase):
    """Test suite for LibraryPolicyRAG retrieval and relevance verification."""

    @classmethod
    def setUpClass(cls):
        cls.rag = get_policy_rag()

    def test_novel_policy_retrieval(self):
        """Case: Retrieve novel rate (₹5/day) and loan period (14 days)."""
        res = self.rag.retrieve_policy("What is the overdue fine for a novel?", book_type="novel")
        self.assertEqual(res.status, "POLICY_FOUND")
        self.assertEqual(res.daily_rate, 5)
        self.assertEqual(res.loan_period, 14)
        self.assertIn("data/library_rules.txt", res.source)

    def test_textbook_policy_retrieval(self):
        """Case: Retrieve textbook rate (₹10/day) and loan period (7 days)."""
        res = self.rag.retrieve_policy("How much is the fine for an overdue textbook?", book_type="textbook")
        self.assertEqual(res.status, "POLICY_FOUND")
        self.assertEqual(res.daily_rate, 10)
        self.assertEqual(res.loan_period, 7)

    def test_reference_policy_retrieval(self):
        """Case: Retrieve reference book rate (₹15/day) and loan period (2 days)."""
        res = self.rag.retrieve_policy("Reference book overdue fine schedule", book_type="reference")
        self.assertEqual(res.status, "POLICY_FOUND")
        self.assertEqual(res.daily_rate, 15)
        self.assertEqual(res.loan_period, 2)

    def test_unknown_policy_rejected(self):
        """Case: Reject unknown rule with explicit POLICY_NOT_FOUND (no hallucinations)."""
        query = "Does the library provide free replacement books for damaged reference books?"
        res = self.rag.retrieve_policy(query)
        self.assertEqual(res.status, "POLICY_NOT_FOUND")
        self.assertIn("Could not find the requested rule", res.message)
        self.assertIn("contact authorized library staff", res.message.lower())

    def test_waiver_policy_retrieval(self):
        """Case: Retrieve waiver approval policy rule (requires staff approval)."""
        res = self.rag.retrieve_policy("Can fine waivers be approved automatically?")
        self.assertEqual(res.status, "POLICY_FOUND")
        self.assertIn("authorized library staff", res.supporting_text.lower())


if __name__ == "__main__":
    unittest.main()
