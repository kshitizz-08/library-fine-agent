"""
LangGraph Supervisor Module with Explicit State & MAX_HOPS Routing (Unit 4)

State Fields:
- user_question: str
- member_id: str
- thread_id: str
- retrieved_policy: dict | None
- policy_status: str ("POLICY_FOUND", "POLICY_NOT_FOUND", "NOT_APPLICABLE")
- book_type: str | None
- overdue_days: int | None
- fine_result: str | None
- current_agent: str
- hop_count: int
- max_hops: int (default 4)
- approval_status: str | None ("PENDING", "APPROVED", "REJECTED", None)
- final_response: str
- execution_trace: list[dict]
- messages: Annotated[list[AnyMessage], add_messages]
"""

import os
import re
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.calculator_agent import CalculatorAgent, calculate_library_fine
from agents.policy_agent import PolicyAgent
from approvals.waiver_approval import WaiverManager, get_waiver_manager
from memory.sqlite_memory import get_sqlite_checkpointer
from rag.rag_engine import get_policy_rag
from reminders.reminder_engine import get_reminder_manager



class SupervisorState(TypedDict):
    """Explicit state schema for Library Supervisor Workflow."""
    user_question: str
    member_id: str
    thread_id: str
    retrieved_policy: Optional[Dict[str, Any]]
    policy_status: str  # "POLICY_FOUND", "POLICY_NOT_FOUND", "NOT_APPLICABLE"
    book_type: Optional[str]
    overdue_days: Optional[int]
    fine_result: Optional[str]
    current_agent: str
    hop_count: int
    max_hops: int
    approval_status: Optional[str]
    final_response: str
    execution_trace: List[Dict[str, Any]]
    intent: str
    messages: Annotated[List[AnyMessage], add_messages]


def extract_book_type(text: str) -> Optional[str]:
    """Extract book classification from text, including standard and non-standard types."""
    t = text.lower()
    if "reference" in t:
        return "reference"
    if "textbook" in t or "text book" in t:
        return "textbook"
    if "novel" in t:
        return "novel"

    # Known unsupported item types
    common_unsupported = ["magazine", "journal", "comic", "dvd", "newspaper", "audiobook", "thesis", "manual"]
    for item in common_unsupported:
        if item in t:
            return item

    # Pattern match for "<category> book" or "fine for a <category>"
    match = re.search(r"\b([a-zA-Z]+)\s+book\b", t)
    if match and match.group(1) not in ["the", "a", "an", "this", "that", "overdue", "damaged", "library"]:
        return match.group(1)

    match2 = re.search(r"(?:for|of)\s+(?:a|an)?\s*([a-zA-Z]+)\s+(?:that|which|was|is|\d)", t)
    if match2 and match2.group(1) not in ["the", "a", "an", "this", "that", "book", "fine", "overdue", "late"]:
        return match2.group(1)

    return None


def extract_overdue_days(text: str) -> Optional[int]:
    """Extract number of overdue days from text, including negative values."""
    # Check explicit patterns like "-3 days", "5 days", "7 days overdue", "returned 5 days late"
    match = re.search(r"(-?\d+)\s*(?:days?|overdue|late)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Match negative numbers explicitly (e.g. "-3")
    neg_match = re.search(r"(-\d+)", text)
    if neg_match:
        return int(neg_match.group(1))

    return None


def search_history_for_days(messages: List[AnyMessage]) -> Optional[int]:
    """Search previous conversation messages for the most recent overdue days value."""
    for msg in reversed(messages):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        days = extract_overdue_days(content)
        if days is not None:
            return days
    return None


def detect_intent(text: str) -> str:
    """Classify the user's intent deterministically."""
    t = text.lower()

    # Due date & Pre-Due Fine Reminder intent
    if any(k in t for k in ["reminder", "remind", "due date", "due soon", "when is my book due", "return date", "books due", "submit the book", "submit book"]):
        return "reminder"

    # Waiver intent
    if any(k in t for k in ["waiver", "waive", "exemption", "waiving", "forgive"]):
        return "waiver"

    # Known unsupported/unknown rules query
    if any(k in t for k in ["replacement", "damaged", "damage", "lost book", "parking", "cafeteria"]):
        return "policy_query"

    # Calculation intent
    if any(k in t for k in ["fine", "overdue", "late", "calculate", "how much", "cost", "and if it were"]):
        return "calculation"

    # General policy/loan period intent
    if any(k in t for k in ["loan period", "period", "policy", "rules", "how long", "borrow"]):
        return "policy_query"

    return "general"


# ============================================================
# GRAPH NODE FUNCTIONS
# ============================================================

def supervisor_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Supervisor entry node:
    - Parses request and checks context/memory for follow-up questions
    - Increments hop count
    - Routes deterministically
    """
    user_q = state.get("user_question", "").strip()
    hop_count = state.get("hop_count", 0) + 1
    max_hops = state.get("max_hops", 4)
    trace = list(state.get("execution_trace", []))

    # Identify book type
    current_book_type = extract_book_type(user_q) or state.get("book_type")

    # Identify overdue days
    current_days = extract_overdue_days(user_q)

    # Unit 3A: Thread Memory - Context reuse for follow-up questions
    # e.g., "And if it were a reference book?" -> reuses previous 5 days overdue
    if current_days is None:
        if state.get("overdue_days") is not None:
            current_days = state.get("overdue_days")
        else:
            hist_days = search_history_for_days(state.get("messages", []))
            if hist_days is not None:
                current_days = hist_days

    intent = detect_intent(user_q)

    # If asking for fine calculation but days are completely missing and no context
    if intent == "calculation" and current_days is None and not any(k in user_q.lower() for k in ["rate", "how much per day"]):
        intent = "clarification_needed"

    trace.append(
        {
            "hop": hop_count,
            "agent": "supervisor",
            "action": "analyze_request",
            "detected_intent": intent,
            "book_type": current_book_type,
            "overdue_days": current_days,
            "current_hop_count": hop_count,
            "max_hops": max_hops,
        }
    )

    return {
        "hop_count": hop_count,
        "book_type": current_book_type,
        "overdue_days": current_days,
        "intent": intent,
        "current_agent": "supervisor",
        "execution_trace": trace,
    }


def policy_agent_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Policy Agent node:
    - Retrieves relevant policy from RAG
    - Never calculates fine amounts, never approves waivers
    """
    hop_count = state.get("hop_count", 0) + 1
    trace = list(state.get("execution_trace", []))
    policy_agent = PolicyAgent()

    user_q = state.get("user_question", "")
    book_type = state.get("book_type")

    rag_result = policy_agent.check_policy(query=user_q, book_type=book_type)
    policy_dict = rag_result.to_dict()

    trace.append(
        {
            "hop": hop_count,
            "agent": "policy_agent",
            "action": "rag_retrieval",
            "status": rag_result.status,
            "source": rag_result.source,
            "daily_rate": rag_result.daily_rate,
            "distance": rag_result.distance,
            "rule_summary": rag_result.rule_summary,
            "current_hop_count": hop_count,
        }
    )

    return {
        "hop_count": hop_count,
        "current_agent": "policy_agent",
        "retrieved_policy": policy_dict,
        "policy_status": rag_result.status,
        "execution_trace": trace,
    }


def calculator_agent_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Calculator Agent node:
    - Performs deterministic calculation using calculate_library_fine tool
    - Uses daily rate from validated policy
    - Never invents rates, never approves waivers
    """
    hop_count = state.get("hop_count", 0) + 1
    trace = list(state.get("execution_trace", []))

    overdue_days = state.get("overdue_days", 0)
    book_type = state.get("book_type") or "unknown"
    policy_dict = state.get("retrieved_policy") or {}
    daily_rate = policy_dict.get("daily_rate")

    calc_agent = CalculatorAgent()
    calc_res = calc_agent.calculate(
        days_overdue=overdue_days if overdue_days is not None else 0,
        book_type=book_type or "novel",
        daily_rate=daily_rate,
    )

    raw_output = calc_res["raw_output"]

    trace.append(
        {
            "hop": hop_count,
            "agent": "calculator_agent",
            "action": "execute_fine_tool",
            "inputs": {
                "days_overdue": overdue_days,
                "book_type": book_type,
                "daily_rate": daily_rate,
            },
            "output": raw_output,
            "fine_amount": calc_res["fine_amount"],
            "current_hop_count": hop_count,
        }
    )

    if calc_res["success"]:
        final_answer = (
            f"{raw_output}\n\n"
            f"📋 **Policy Evidence Verified:**\n"
            f"- Source: `{policy_dict.get('source', 'data/library_rules.txt')}`\n"
            f"- Rate applied: ₹{daily_rate or calc_res.get('daily_rate')}/day."
        )
    else:
        final_answer = f"⚠️ Calculation Error: {calc_res['error']}"

    return {
        "hop_count": hop_count,
        "current_agent": "calculator_agent",
        "fine_result": raw_output,
        "final_response": final_answer,
        "execution_trace": trace,
        "messages": [AIMessage(content=final_answer)],
    }


def waiver_handler_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Waiver Handler node:
    - Creates a PENDING waiver request in SQLite
    - Enforces human-in-the-loop: waiver is never granted automatically
    - Attaches retrieved waiver policy evidence
    """
    hop_count = state.get("hop_count", 0) + 1
    trace = list(state.get("execution_trace", []))

    member_id = state.get("member_id", "MEMBER_GUEST")
    thread_id = state.get("thread_id", "THREAD_DEFAULT")
    book_type = state.get("book_type") or "novel"
    days_overdue = state.get("overdue_days") or 5
    policy_dict = state.get("retrieved_policy") or {}

    daily_rate = policy_dict.get("daily_rate") or (5 if book_type == "novel" else (10 if book_type == "textbook" else 15))
    fine_amount = max(0, days_overdue) * daily_rate

    waiver_mgr = get_waiver_manager()
    waiver_record = waiver_mgr.create_waiver_request(
        member_id=member_id,
        thread_id=thread_id,
        book_type=book_type,
        days_overdue=days_overdue,
        fine_amount=fine_amount,
        reason=state.get("user_question", "Fine waiver requested"),
        policy_evidence=policy_dict.get("supporting_text", "Fine waivers require staff review."),
    )

    req_id = waiver_record["request_id"]
    status = waiver_record["status"]  # "PENDING"

    trace.append(
        {
            "hop": hop_count,
            "agent": "waiver_handler",
            "action": "create_pending_waiver",
            "request_id": req_id,
            "status": status,
            "fine_amount": fine_amount,
            "current_hop_count": hop_count,
        }
    )

    response_text = (
        f"📝 **Fine Waiver Request Submitted**\n\n"
        f"- **Request ID:** `{req_id}`\n"
        f"- **Status:** ⏳ **PENDING STAFF REVIEW**\n"
        f"- **Member ID:** `{member_id}`\n"
        f"- **Book Category:** {book_type.capitalize()} ({days_overdue} days overdue)\n"
        f"- **Calculated Fine:** ₹{fine_amount}\n\n"
        f"🔒 **Policy Rule (Section 3):** Under university library rules, automated systems "
        f"cannot grant waivers. Your request is queued in PENDING status and requires review "
        f"by authorized library staff in the staff portal."
    )

    return {
        "hop_count": hop_count,
        "current_agent": "waiver_handler",
        "approval_status": "PENDING",
        "final_response": response_text,
        "execution_trace": trace,
        "messages": [AIMessage(content=response_text)],
    }


def policy_not_found_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Policy Not Found terminal node:
    - Stops calculation path when policy is missing
    - Strictly avoids hallucinating or inventing rules
    """
    hop_count = state.get("hop_count", 0) + 1
    trace = list(state.get("execution_trace", []))
    policy_dict = state.get("retrieved_policy") or {}

    msg = policy_dict.get("message") or (
        "Could not find the requested rule in the available library policy document. "
        "Please contact authorized library staff."
    )

    response_text = (
        f"❌ **Policy Rule Not Found (POLICY_NOT_FOUND)**\n\n"
        f"{msg}\n\n"
        f"📌 *Note: The system strictly avoids inventing unlisted policies. "
        f"No fine calculation or waiver was executed.*"
    )

    trace.append(
        {
            "hop": hop_count,
            "agent": "policy_agent",
            "action": "reject_unsupported_query",
            "status": "POLICY_NOT_FOUND",
            "reason": "Rule not documented in library_rules.txt",
            "current_hop_count": hop_count,
        }
    )

    return {
        "hop_count": hop_count,
        "current_agent": "policy_agent",
        "final_response": response_text,
        "execution_trace": trace,
        "messages": [AIMessage(content=response_text)],
    }


def policy_answer_node(state: SupervisorState) -> Dict[str, Any]:
    """Node for answering general policy questions from retrieved evidence."""
    hop_count = state.get("hop_count", 0) + 1
    trace = list(state.get("execution_trace", []))
    policy_dict = state.get("retrieved_policy") or {}

    response_text = (
        f"📖 **Library Policy Information:**\n\n"
        f"{policy_dict.get('supporting_text', policy_dict.get('rule_summary'))}\n\n"
        f"**Source:** `{policy_dict.get('source', 'data/library_rules.txt')}`"
    )

    trace.append(
        {
            "hop": hop_count,
            "agent": "policy_agent",
            "action": "answer_policy_query",
            "current_hop_count": hop_count,
        }
    )

    return {
        "hop_count": hop_count,
        "current_agent": "policy_agent",
        "final_response": response_text,
        "execution_trace": trace,
        "messages": [AIMessage(content=response_text)],
    }


def clarification_node(state: SupervisorState) -> Dict[str, Any]:
    """Node asking user for missing information when context is not available."""
    hop_count = state.get("hop_count", 0) + 1
    trace = list(state.get("execution_trace", []))

    response_text = (
        "❓ I see you would like to check a library fine, but I need the number of overdue days. "
        "Could you please specify how many days overdue the book is?"
    )

    trace.append(
        {
            "hop": hop_count,
            "agent": "supervisor",
            "action": "request_clarification",
            "current_hop_count": hop_count,
        }
    )

    return {
        "hop_count": hop_count,
        "current_agent": "supervisor",
        "final_response": response_text,
        "execution_trace": trace,
        "messages": [AIMessage(content=response_text)],
    }


def reminder_handler_node(state: SupervisorState) -> Dict[str, Any]:
    """Node for checking active borrower loans and issuing pre-due date reminders."""
    hop_count = state.get("hop_count", 0) + 1
    trace = list(state.get("execution_trace", []))
    member_id = state.get("member_id", "MEM-001")
    rm = get_reminder_manager()
    loans = rm.get_member_loans(member_id, status_filter="ACTIVE")

    if not loans:
        resp = f"**Borrower Record (`{member_id}`):** You currently have no active borrowed books on record."
    else:
        lines = [
            f"🔔 **Due Date & Pre-Due Fine Reminders for Member `{member_id}`:**\n",
            "Under Library Borrowing Policy (Section 1 & 2), all loaned materials must be returned on or before their scheduled due date. Late returns incur mandatory daily overdue fines.\n",
        ]
        for idx, loan in enumerate(loans, 1):
            urgency = loan.get("urgency", "ON_TRACK")
            title = loan.get("book_title")
            b_type = loan.get("book_type", "").capitalize()
            due = loan.get("due_date")
            rate = loan.get("daily_rate")
            remaining = loan.get("days_remaining", 0)

            if urgency == "OVERDUE":
                days_over = abs(remaining)
                fine = loan.get("fine_accrued", 0)
                lines.append(
                    f"{idx}. **{title}** ({b_type}) — 🚨 **OVERDUE by {days_over} day(s)** (Due was: {due})\n"
                    f"   - **Current Fine Liability:** ₹{fine} (accumulating at ₹{rate}/day)\n"
                    f"   - **Action Required:** Return book immediately to circulation desk to halt further fines."
                )
            elif urgency == "DUE_TODAY":
                lines.append(
                    f"{idx}. **{title}** ({b_type}) — ⚠️ **DUE TODAY** ({due})\n"
                    f"   - **Pre-Due Warning:** Return today before desk closes. A fine of ₹{rate}/day will begin accruing tomorrow.\n"
                    f"   - **Action Required:** Return to desk today."
                )
            elif urgency == "DUE_SOON":
                lines.append(
                    f"{idx}. **{title}** ({b_type}) — 🔔 **Due in {remaining} day(s)** ({due})\n"
                    f"   - **Pre-Due Warning:** Please submit on or before the due date, otherwise a late fine of ₹{rate}/day will be charged.\n"
                    f"   - **Action Required:** Return on or before {due}."
                )
            else:
                lines.append(
                    f"{idx}. **{title}** ({b_type}) — 🟢 Due in {remaining} day(s) ({due})\n"
                    f"   - **Status:** Healthy loan. Daily overdue fine if returned late: ₹{rate}/day."
                )

        resp = "\n".join(lines)

    trace.append(
        {
            "hop": hop_count,
            "agent": "reminder_agent",
            "action": "evaluate_due_date_reminders",
            "member_id": member_id,
            "active_loans": len(loans),
            "current_hop_count": hop_count,
        }
    )

    return {
        "hop_count": hop_count,
        "current_agent": "reminder_agent",
        "final_response": resp,
        "execution_trace": trace,
        "messages": [AIMessage(content=resp)],
    }


def max_hops_reached_node(state: SupervisorState) -> Dict[str, Any]:
    """Terminal node when MAX_HOPS limit is reached."""
    max_hops = state.get("max_hops", 4)
    response_text = (
        f"🛑 **Routing Halted (MAX_HOPS_REACHED)**: Execution exceeded maximum allowed hops ({max_hops}) "
        f"to prevent unbounded routing loops."
    )
    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "hop": state.get("hop_count", max_hops),
            "agent": "supervisor",
            "action": "halt_max_hops",
            "reason": f"Hop count exceeded MAX_HOPS ({max_hops})",
        }
    )
    return {
        "final_response": response_text,
        "execution_trace": trace,
        "messages": [AIMessage(content=response_text)],
    }


# ============================================================
# CONDITIONAL ROUTING EDGES
# ============================================================

def route_from_supervisor(state: SupervisorState) -> str:
    """Deterministic routing from supervisor node."""
    if state.get("hop_count", 0) >= state.get("max_hops", 4):
        return "max_hops_reached"

    intent = state.get("intent", "calculation")
    if intent == "reminder":
        return "reminder_handler"

    if intent == "clarification_needed":
        return "clarification"

    # All policy, calculation, and waiver inquiries route first to Policy Agent
    return "policy_agent"


def route_from_policy(state: SupervisorState) -> str:
    """Deterministic routing from policy agent node."""
    if state.get("hop_count", 0) >= state.get("max_hops", 4):
        return "max_hops_reached"

    policy_status = state.get("policy_status")
    intent = state.get("intent", "calculation")

    # If policy is not found, halt calculation immediately and explain
    if policy_status == "POLICY_NOT_FOUND":
        return "policy_not_found"

    if intent == "waiver":
        return "waiver_handler"

    if intent == "policy_query":
        return "policy_answer"

    # Normal calculation path
    return "calculator_agent"


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_supervisor_graph():
    """
    Construct the LangGraph StateGraph with explicit state,
    deterministic routing, and MAX_HOPS cutoff.
    """
    workflow = StateGraph(SupervisorState)

    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("reminder_handler", reminder_handler_node)
    workflow.add_node("policy_agent", policy_agent_node)
    workflow.add_node("calculator_agent", calculator_agent_node)
    workflow.add_node("waiver_handler", waiver_handler_node)
    workflow.add_node("policy_not_found", policy_not_found_node)
    workflow.add_node("policy_answer", policy_answer_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("max_hops_reached", max_hops_reached_node)

    # Add Edges
    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "reminder_handler": "reminder_handler",
            "policy_agent": "policy_agent",
            "clarification": "clarification",
            "max_hops_reached": "max_hops_reached",
        },
    )

    workflow.add_conditional_edges(
        "policy_agent",
        route_from_policy,
        {
            "calculator_agent": "calculator_agent",
            "waiver_handler": "waiver_handler",
            "policy_answer": "policy_answer",
            "policy_not_found": "policy_not_found",
            "max_hops_reached": "max_hops_reached",
        },
    )

    workflow.add_edge("calculator_agent", END)
    workflow.add_edge("waiver_handler", END)
    workflow.add_edge("policy_not_found", END)
    workflow.add_edge("policy_answer", END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("reminder_handler", END)
    workflow.add_edge("max_hops_reached", END)

    return workflow



class LibrarySupervisor:
    """
    Compiled Supervisor service with SQLite thread persistence.
    """

    def __init__(self, checkpointer=None, max_hops: int = 4):
        self.max_hops = max_hops
        self.checkpointer = checkpointer or get_sqlite_checkpointer()
        self.workflow = build_supervisor_graph()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def invoke(
        self,
        question: str,
        member_id: str = "MEM-001",
        thread_id: str = "thread-default",
        max_hops: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the supervisor workflow for a conversation turn.
        """
        hops = max_hops if max_hops is not None else self.max_hops

        initial_state = {
            "user_question": question,
            "member_id": member_id,
            "thread_id": thread_id,
            "retrieved_policy": None,
            "policy_status": "NOT_APPLICABLE",
            "book_type": None,
            "overdue_days": None,
            "fine_result": None,
            "current_agent": "start",
            "hop_count": 0,
            "max_hops": hops,
            "approval_status": None,
            "final_response": "",
            "execution_trace": [],
            "intent": "",
            "messages": [HumanMessage(content=question)],
        }

        config = {"configurable": {"thread_id": str(thread_id)}}
        result = self.app.invoke(initial_state, config=config)
        return result
