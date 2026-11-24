"""
Library Fine Agent — Baseline & Upgraded Supervisor Integration

Provides:
- calculate_library_fine (@tool)
- PolicyAgent, CalculatorAgent, and LibrarySupervisor
- Backward-compatible `agent` object for legacy callers
- Standalone CLI execution tests with trace logging
"""

import sys
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from agents.calculator_agent import CalculatorAgent, calculate_library_fine
from agents.policy_agent import PolicyAgent
from agents.supervisor import LibrarySupervisor, build_supervisor_graph
from memory.sqlite_memory import MemberProfileStore, get_sqlite_checkpointer
from rag.rag_engine import get_policy_rag

# Ensure stdout handles UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class SupervisorAgentAdapter:
    """
    Backward-compatibility adapter for legacy code calling:
    agent.invoke({"messages": [("user", question)]})
    """

    def __init__(self, supervisor: Optional[LibrarySupervisor] = None):
        self.supervisor = supervisor or LibrarySupervisor(max_hops=4)

    def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Translate legacy inputs to supervisor workflow and return legacy format.
        """
        messages = inputs.get("messages", [])
        question = ""
        for m in messages:
            if isinstance(m, tuple) and len(m) >= 2:
                if m[0] == "user":
                    question = str(m[1])
            elif hasattr(m, "content"):
                question = str(m.content)
            elif isinstance(m, str):
                question = m

        thread_id = "default-session"
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id", thread_id)

        result = self.supervisor.invoke(
            question=question,
            member_id="MEM-001",
            thread_id=thread_id,
        )

        final_content = result.get("final_response", "")

        # Format output messages
        out_messages = [
            HumanMessage(content=question),
            AIMessage(content=final_content),
        ]

        return {
            "messages": out_messages,
            "supervisor_result": result,
            "final_response": final_content,
            "execution_trace": result.get("execution_trace", []),
            "policy_status": result.get("policy_status", "NOT_APPLICABLE"),
            "hop_count": result.get("hop_count", 0),
        }


# Global instances
supervisor = LibrarySupervisor(max_hops=4)
agent = SupervisorAgentAdapter(supervisor)


def print_trace(result: Dict[str, Any]) -> None:
    """Pretty-print execution trace for CLI tests."""
    print("\n" + "=" * 70)
    print("SUPERVISOR & AGENT EXECUTION TRACE")
    print("=" * 70)

    trace = result.get("execution_trace", [])
    if trace:
        for step in trace:
            hop = step.get("hop", "?")
            agent_name = step.get("agent", "unknown").upper()
            action = step.get("action", "")
            print(f"\n[HOP {hop}] AGENT: {agent_name} | ACTION: {action}")
            print("-" * 70)
            for k, v in step.items():
                if k not in ("hop", "agent", "action"):
                    print(f"  {k}: {v}")
    else:
        messages = result.get("messages", [])
        for i, m in enumerate(messages, start=1):
            print(f"\n[{i}] {type(m).__name__}: {getattr(m, 'content', '')}")

    print("\n" + "=" * 70)
    print(f"FINAL RESPONSE:\n{result.get('final_response', '')}")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("TEST 1: SUCCESSFUL FINE CALCULATION (Novel, 5 days overdue)")
    print("#" * 70)
    t1_q = "A novel was returned 5 days late. How much is the library fine?"
    res1 = supervisor.invoke(t1_q, member_id="MEM-001", thread_id="cli-thread-1")
    print_trace(res1)

    print("\n" + "#" * 70)
    print("TEST 2: THREAD MEMORY FOLLOW-UP (Reference book, reuse 5 days)")
    print("#" * 70)
    t2_q = "And if it were a reference book?"
    res2 = supervisor.invoke(t2_q, member_id="MEM-001", thread_id="cli-thread-1")
    print_trace(res2)

    print("\n" + "#" * 70)
    print("TEST 3: UNKNOWN POLICY RULE (Free replacement for damaged books)")
    print("#" * 70)
    t3_q = "Does the library provide free replacement books for damaged reference books?"
    res3 = supervisor.invoke(t3_q, member_id="MEM-001", thread_id="cli-thread-2")
    print_trace(res3)

    print("\n" + "#" * 70)
    print("TEST 4: DELIBERATE ERROR (Negative overdue days)")
    print("#" * 70)
    t4_q = "Calculate the fine for a novel that is -3 days overdue."
    res4 = supervisor.invoke(t4_q, member_id="MEM-001", thread_id="cli-thread-3")
    print_trace(res4)

    print("\n" + "#" * 70)
    print("TEST 5: FINE WAIVER REQUEST (Hospital admission)")
    print("#" * 70)
    t5_q = "I need a fine waiver for my overdue textbook because I was in the hospital."
    res5 = supervisor.invoke(t5_q, member_id="MEM-001", thread_id="cli-thread-4")
    print_trace(res5)