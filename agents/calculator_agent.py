"""
Calculator Agent Module (Section 5)

Responsibilities:
- Use the single calculate_library_fine Python @tool.
- Perform deterministic arithmetic: fine = days_overdue * daily_rate.
- Validate overdue days (non-negative) and book type.
- Use the daily rate supplied by validated policy result.
- Return amount and plain-language explanation.
- Never invent a daily rate.
- Never make policy decisions or approve waivers.
"""

from typing import Any, Dict, Optional
from langchain_core.tools import tool


@tool
def calculate_library_fine(
    days_overdue: int,
    book_type: str,
    daily_rate: Optional[int] = None,
) -> str:
    """
    Calculate the library fine based on the number of overdue days and book category.
    Performs deterministic arithmetic using the rate provided by validated policy.

    Args:
        days_overdue: Number of days the book is overdue.
        book_type: Type of book: novel, textbook, or reference.
        daily_rate: Optional rate passed from validated policy retrieval.

    Returns:
        Fine amount and explanation, or an error message if inputs are invalid.
    """
    if days_overdue < 0:
        return "Error: days_overdue cannot be negative."

    standard_rates = {
        "novel": 5,
        "textbook": 10,
        "reference": 15,
    }

    clean_type = str(book_type).lower().strip()
    if clean_type not in standard_rates:
        return (
            f"Error: Unknown book_type '{book_type}'. "
            "Valid types are novel, textbook, and reference."
        )

    # Use policy rate if supplied, else fallback to standard rate
    rate = int(daily_rate) if daily_rate is not None else standard_rates[clean_type]
    fine = days_overdue * rate

    return (
        f"Fine: ₹{fine}. "
        f"A {clean_type} has a fine of ₹{rate} per overdue day "
        f"for {days_overdue} overdue days."
    )


class CalculatorAgent:
    """
    Deterministic fine calculation agent.
    Never invents rates, never retrieves policy, never approves waivers.
    """

    def __init__(self):
        self.tool = calculate_library_fine

    def calculate(
        self,
        days_overdue: int,
        book_type: str,
        daily_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute deterministic calculation and return structured output.
        """
        output_str = self.tool.invoke(
            {
                "days_overdue": days_overdue,
                "book_type": book_type,
                "daily_rate": daily_rate,
            }
        )

        is_error = output_str.startswith("Error:")
        amount = None
        if not is_error and "Fine: ₹" in output_str:
            try:
                part = output_str.split("Fine: ₹")[1].split(".")[0]
                amount = float(part)
            except Exception:
                amount = None

        return {
            "success": not is_error,
            "raw_output": output_str,
            "fine_amount": amount,
            "days_overdue": days_overdue,
            "book_type": book_type,
            "daily_rate": daily_rate,
            "error": output_str if is_error else None,
        }
