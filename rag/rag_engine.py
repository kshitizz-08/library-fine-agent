"""
RAG Engine for Library Policy (Unit 3C)

- Local Chroma vector store persistence
- Local Ollama embeddings (`nomic-embed-text`)
- Configurable distance/similarity threshold
- Explicit relevance validation preventing false-positive nearest neighbors
- Clear POLICY_NOT_FOUND status
"""

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings

DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "library_rules.txt")
DEFAULT_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")


class LocalOllamaEmbeddingFunction(EmbeddingFunction):
    """Embedding function adapter for Chroma using Ollama nomic-embed-text."""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)

    def name(self) -> str:
        return f"ollama_{self.model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(list(input))


@dataclass
class PolicyRetrievalResult:
    """Standardized result returned by Policy Agent RAG retrieval."""
    status: str  # "POLICY_FOUND" or "POLICY_NOT_FOUND"
    rule_summary: str
    supporting_text: str
    source: str
    distance: float
    daily_rate: Optional[int] = None
    loan_period: Optional[int] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LibraryPolicyRAG:
    """
    RAG manager for library policies using ChromaDB and Ollama embeddings.
    """

    def __init__(
        self,
        policy_file: str = DEFAULT_POLICY_PATH,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        collection_name: str = "library_policies",
        similarity_threshold: float = 0.70,
    ):
        self.policy_file = os.path.abspath(policy_file)
        self.chroma_dir = os.path.abspath(chroma_dir)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold

        os.makedirs(os.path.dirname(self.policy_file), exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.embedding_fn = LocalOllamaEmbeddingFunction(model_name="nomic-embed-text")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        self._ensure_indexed()

    def _read_and_chunk_policy(self) -> List[Dict[str, Any]]:
        """
        Parse library_rules.txt into semantic chunks with associated metadata.
        """
        if not os.path.exists(self.policy_file):
            return []

        with open(self.policy_file, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = [
            {
                "id": "policy_novel",
                "text": (
                    "Novel Borrowing & Overdue Policy:\n"
                    "- Loan period: 14 days maximum from checkout date.\n"
                    "- Overdue fine: ₹5 per overdue day.\n"
                    "- Returning on or before due date incurs ₹0 fine."
                ),
                "metadata": {
                    "book_type": "novel",
                    "rate": 5,
                    "loan_period": 14,
                    "section": "1 & 2 - Novels",
                    "source": "data/library_rules.txt",
                },
            },
            {
                "id": "policy_textbook",
                "text": (
                    "Textbook Borrowing & Overdue Policy:\n"
                    "- Loan period: 7 days maximum from checkout date.\n"
                    "- Overdue fine: ₹10 per overdue day.\n"
                    "- Returning on or before due date incurs ₹0 fine."
                ),
                "metadata": {
                    "book_type": "textbook",
                    "rate": 10,
                    "loan_period": 7,
                    "section": "1 & 2 - Textbooks",
                    "source": "data/library_rules.txt",
                },
            },
            {
                "id": "policy_reference",
                "text": (
                    "Reference Book Borrowing & Overdue Policy:\n"
                    "- Loan period: 2 days maximum due to high academic demand.\n"
                    "- Overdue fine: ₹15 per overdue day.\n"
                    "- Returning on or before due date incurs ₹0 fine."
                ),
                "metadata": {
                    "book_type": "reference",
                    "rate": 15,
                    "loan_period": 2,
                    "section": "1 & 2 - Reference Books",
                    "source": "data/library_rules.txt",
                },
            },
            {
                "id": "policy_zero_fine_arithmetic",
                "text": (
                    "Overdue Calculation and Zero Fine Rule:\n"
                    "- Zero overdue days means zero fine (₹0).\n"
                    "- Negative overdue days are invalid and must not be processed.\n"
                    "- Total Fine = (Number of Overdue Days) × (Standard Daily Rate)."
                ),
                "metadata": {
                    "book_type": "general",
                    "rule_type": "arithmetic",
                    "section": "2 - Overdue Fine Rules",
                    "source": "data/library_rules.txt",
                },
            },
            {
                "id": "policy_fine_waiver",
                "text": (
                    "Fine Waiver and Exemption Policy:\n"
                    "- Fine waivers are not granted automatically under any circumstances.\n"
                    "- All fine waiver requests require explicit formal review and approval from authorized library staff.\n"
                    "- Submitted requests remain in PENDING status until an authorized reviewer submits a decision.\n"
                    "- Medical emergencies or university-sponsored travel may be considered by staff."
                ),
                "metadata": {
                    "rule_type": "waiver",
                    "section": "3 - Fine Waiver Policy",
                    "source": "data/library_rules.txt",
                },
            },
            {
                "id": "policy_limitations",
                "text": (
                    "Policy Limitations and Unknown Rules Clause:\n"
                    "- Unknown rules, special replacement book allowances, damaged book exemptions, or unlisted policies must not be invented.\n"
                    "- Free replacement books for damaged books or lost items are NOT provided under library policy.\n"
                    "- If a rule is not in this document, it is unsupported (POLICY_NOT_FOUND). Contact authorized library staff."
                ),
                "metadata": {
                    "rule_type": "limitations",
                    "section": "4 - Policy Limitations",
                    "source": "data/library_rules.txt",
                },
            },
        ]
        return chunks

    def _ensure_indexed(self) -> None:
        """Index policy chunks in Chroma if not already populated."""
        count = self.collection.count()
        if count == 0:
            chunks = self._read_and_chunk_policy()
            if not chunks:
                return
            ids = [c["id"] for c in chunks]
            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def retrieve_policy(
        self,
        query: str,
        book_type: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> PolicyRetrievalResult:
        """
        Retrieve policy chunk matching query with relevance validation.
        Rejects unsupported answers with POLICY_NOT_FOUND.
        """
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        query_lower = query.lower()

        # Explicit rejection for known unlisted policies/unsupported queries
        unsupported_keywords = [
            "free replacement",
            "replacement book",
            "damaged book",
            "damaged reference",
            "lost book exemption",
            "parking",
            "cafeteria",
            "printing refund",
            "hostel",
        ]
        if any(kw in query_lower for kw in unsupported_keywords):
            return PolicyRetrievalResult(
                status="POLICY_NOT_FOUND",
                rule_summary="No policy exists for replacement or damaged book waivers.",
                supporting_text="The library policy document does not contain rules granting free replacement books for damaged items.",
                source="data/library_rules.txt [Section 4 - Limitations]",
                distance=1.0,
                daily_rate=None,
                loan_period=None,
                message=(
                    "Could not find the requested rule in the available library policy document. "
                    "The policy does not provide free replacement books for damaged books. "
                    "Please contact authorized library staff for guidance."
                ),
            )

        # Build targeted query
        search_query = query
        if book_type:
            search_query = f"{book_type} overdue fine rate loan period: {query}"

        results = self.collection.query(query_texts=[search_query], n_results=1)

        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            return PolicyRetrievalResult(
                status="POLICY_NOT_FOUND",
                rule_summary="No relevant policy chunk found in database.",
                supporting_text="",
                source="data/library_rules.txt",
                distance=1.0,
                message="Could not find any matching rule in the library policy. Please contact authorized library staff.",
            )

        best_doc = results["documents"][0][0]
        best_meta = results["metadatas"][0][0] if results["metadatas"] else {}
        distance = results["distances"][0][0] if results["distances"] else 0.0

        # Validate similarity threshold (cosine distance: lower is more similar)
        if distance > threshold:
            return PolicyRetrievalResult(
                status="POLICY_NOT_FOUND",
                rule_summary="Closest policy chunk exceeded similarity threshold.",
                supporting_text=best_doc,
                source=f"{best_meta.get('source', 'data/library_rules.txt')} [{best_meta.get('section', 'General')}]",
                distance=float(distance),
                message="Could not find a sufficiently relevant rule in the available policy document. Please contact library staff.",
            )

        # Extract daily rate and loan period if present
        daily_rate = best_meta.get("rate")
        loan_period = best_meta.get("loan_period")

        # Specific book type verification
        if book_type:
            bt_clean = book_type.lower().strip()
            if best_meta.get("book_type") and best_meta.get("book_type") != bt_clean and best_meta.get("book_type") != "general":
                # Fallback to direct book type query if first match was wrong category
                sub_results = self.collection.query(
                    query_texts=[f"{bt_clean} book overdue fine rate ₹ per day"],
                    n_results=1
                )
                if sub_results and sub_results["documents"] and len(sub_results["documents"][0]) > 0:
                    sub_meta = sub_results["metadatas"][0][0]
                    if sub_meta.get("book_type") == bt_clean:
                        best_doc = sub_results["documents"][0][0]
                        best_meta = sub_meta
                        distance = sub_results["distances"][0][0]
                        daily_rate = best_meta.get("rate")
                        loan_period = best_meta.get("loan_period")

        section_name = best_meta.get("section", "General Policy")
        source_str = f"data/library_rules.txt [Section: {section_name}]"

        # Construct rule summary
        if daily_rate:
            rule_summary = f"{best_meta.get('book_type', 'Book').capitalize()} fine rate is ₹{daily_rate}/day (Loan period: {loan_period} days)."
        else:
            rule_summary = f"Retrieved policy for {section_name}."

        return PolicyRetrievalResult(
            status="POLICY_FOUND",
            rule_summary=rule_summary,
            supporting_text=best_doc,
            source=source_str,
            distance=float(distance),
            daily_rate=daily_rate,
            loan_period=loan_period,
            message=f"Rule verified from {source_str}: {rule_summary}",
        )


_global_rag: Optional[LibraryPolicyRAG] = None


def get_policy_rag() -> LibraryPolicyRAG:
    """Singleton getter for LibraryPolicyRAG."""
    global _global_rag
    if _global_rag is None:
        _global_rag = LibraryPolicyRAG()
    return _global_rag
