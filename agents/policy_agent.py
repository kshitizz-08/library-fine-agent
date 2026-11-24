"""
Policy Agent Module (Section 5)

Responsibilities:
- Retrieve relevant library policy through RAG.
- Return the rule, supporting text, source, and retrieval status.
- Handle missing policy with POLICY_NOT_FOUND.
- Never calculate authoritative fine amounts.
- Never approve waivers.
"""

from typing import Any, Dict, Optional
from rag.rag_engine import LibraryPolicyRAG, PolicyRetrievalResult, get_policy_rag


class PolicyAgent:
    """
    Library policy verification agent powered by RAG.
    Strictly reports policy facts and rejects unknown rules.
    Never calculates fines and never approves waivers.
    """

    def __init__(self, rag_engine: Optional[LibraryPolicyRAG] = None):
        self.rag = rag_engine or get_policy_rag()

    def check_policy(
        self,
        query: str,
        book_type: Optional[str] = None,
    ) -> PolicyRetrievalResult:
        """
        Verify library policy for a given user query and optional book category.
        """
        return self.rag.retrieve_policy(query=query, book_type=book_type)

    def get_rate_for_book(self, book_type: str) -> PolicyRetrievalResult:
        """
        Retrieve validated overdue rate for a specific book category.
        """
        clean_type = book_type.lower().strip()
        query = f"What is the overdue fine rate for a {clean_type}?"
        return self.rag.retrieve_policy(query=query, book_type=clean_type)
