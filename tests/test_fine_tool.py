"""
Unit tests for the calculate_library_fine tool (Section 5 & 9).
Tests deterministic arithmetic, zero days, negative days, unknown types, and rate overrides.
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.calculator_agent import CalculatorAgent, calculate_library_fine


class TestFineTool(unittest.TestCase):
    """Test suite for calculate_library_fine tool and CalculatorAgent."""

    def setUp(self):
        self.agent = CalculatorAgent()

    def test_novel_calculation(self):
        """Case: Novel overdue by 5 days -> ₹25."""
        out = calculate_library_fine.invoke({"days_overdue": 5, "book_type": "novel"})
        self.assertIn("Fine: ₹25", out)
        self.assertIn("₹5 per overdue day", out)

    def test_textbook_calculation(self):
        """Case: Textbook overdue by 4 days -> ₹40."""
        out = calculate_library_fine.invoke({"days_overdue": 4, "book_type": "textbook"})
        self.assertIn("Fine: ₹40", out)
        self.assertIn("₹10 per overdue day", out)

    def test_reference_calculation(self):
        """Case: Reference book overdue by 3 days -> ₹45."""
        out = calculate_library_fine.invoke({"days_overdue": 3, "book_type": "reference"})
        self.assertIn("Fine: ₹45", out)
        self.assertIn("₹15 per overdue day", out)

    def test_zero_days_overdue(self):
        """Case: Returned on time (0 days overdue) -> ₹0 fine."""
        out = calculate_library_fine.invoke({"days_overdue": 0, "book_type": "novel"})
        self.assertIn("Fine: ₹0", out)

    def test_negative_days_error(self):
        """Case: Negative overdue days (-3) -> validation error."""
        out = calculate_library_fine.invoke({"days_overdue": -3, "book_type": "novel"})
        self.assertEqual(out, "Error: days_overdue cannot be negative.")

    def test_unknown_book_type_error(self):
        """Case: Unknown book category -> validation error."""
        out = calculate_library_fine.invoke({"days_overdue": 5, "book_type": "magazine"})
        self.assertIn("Error: Unknown book_type 'magazine'", out)
        self.assertIn("Valid types are novel, textbook, and reference", out)

    def test_policy_rate_override(self):
        """Case: Rate explicitly supplied by Policy Agent is used."""
        out = calculate_library_fine.invoke({"days_overdue": 5, "book_type": "novel", "daily_rate": 8})
        self.assertIn("Fine: ₹40", out)
        self.assertIn("₹8 per overdue day", out)

    def test_calculator_agent_structured_output(self):
        """Case: CalculatorAgent wrapper returns structured dictionary."""
        res = self.agent.calculate(days_overdue=5, book_type="novel", daily_rate=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["fine_amount"], 25.0)
        self.assertIsNone(res["error"])


if __name__ == "__main__":
    unittest.main()
