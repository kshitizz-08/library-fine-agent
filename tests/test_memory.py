"""
Unit tests for Thread Memory and SQLite Member Profiles (Unit 3A & 3B).
"""

import os
import sys
import tempfile
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.supervisor import LibrarySupervisor
from memory.sqlite_memory import MemberProfileStore, close_checkpoint_connection, get_sqlite_checkpointer


class TestMemory(unittest.TestCase):
    """Test suite for MemberProfileStore and LangGraph SQLite Thread Memory."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.profile_db = os.path.join(self.temp_dir.name, "profiles.db")
        self.checkpoint_db = os.path.join(self.temp_dir.name, "checkpoints.db")
        self.store = MemberProfileStore(db_path=self.profile_db)

    def tearDown(self):
        close_checkpoint_connection()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_profile_create_and_retrieve(self):
        """Unit 3B: Create and retrieve profile using (member_id, 'profile')."""
        data = {"name": "Bob Smith", "department": "Computer Science", "tier": "Undergraduate"}
        self.store.save_profile("MEM-101", data)
        retrieved = self.store.get_profile("MEM-101")
        self.assertEqual(retrieved["name"], "Bob Smith")
        self.assertEqual(retrieved["department"], "Computer Science")
        self.assertEqual(retrieved["tier"], "Undergraduate")

    def test_profile_update_preserves_unrelated_fields(self):
        """Unit 3B: Update profile without losing existing unrelated keys."""
        initial = {"name": "Charlie", "email": "charlie@college.edu", "membership": "active"}
        self.store.save_profile("MEM-202", initial)

        # Update only department
        self.store.update_profile("MEM-202", {"department": "Physics"})

        updated = self.store.get_profile("MEM-202")
        self.assertEqual(updated["department"], "Physics")
        self.assertEqual(updated["name"], "Charlie")
        self.assertEqual(updated["email"], "charlie@college.edu")
        self.assertEqual(updated["membership"], "active")

    def test_profile_persistence_after_reconnect(self):
        """Unit 3B: Demonstrate profile persists across simulated application restarts."""
        self.store.save_profile("MEM-303", {"name": "Dana", "books_checked_out": 2})

        # Reopen database using a brand new store instance
        reopened_store = MemberProfileStore(db_path=self.profile_db)
        persisted = reopened_store.get_profile("MEM-303")
        self.assertEqual(persisted["name"], "Dana")
        self.assertEqual(persisted["books_checked_out"], 2)

    def test_thread_memory_followup_context_reuse(self):
        """Unit 3A: Follow-up question reuses overdue days from previous turn in same thread."""
        checkpointer = get_sqlite_checkpointer(self.checkpoint_db)
        supervisor = LibrarySupervisor(checkpointer=checkpointer)

        thread_id = "mem-test-thread-alpha"

        # Turn 1: 5 days overdue novel -> ₹25
        res1 = supervisor.invoke("My novel is 5 days overdue. Calculate my fine.", thread_id=thread_id)
        self.assertIn("Fine: ₹25", res1["final_response"])
        self.assertEqual(res1["overdue_days"], 5)

        # Turn 2: Follow-up without repeating overdue days
        res2 = supervisor.invoke("And if it were a reference book?", thread_id=thread_id)
        self.assertIn("Fine: ₹75", res2["final_response"])
        self.assertEqual(res2["book_type"], "reference")
        self.assertEqual(res2["overdue_days"], 5)

    def test_thread_isolation(self):
        """Unit 3A: Verify different conversation threads remain isolated."""
        checkpointer = get_sqlite_checkpointer(self.checkpoint_db)
        supervisor = LibrarySupervisor(checkpointer=checkpointer)

        # Thread 1 has context of 5 days overdue
        supervisor.invoke("A novel is 5 days late.", thread_id="thread-isolated-1")

        # Thread 2 asks follow-up with no prior context in Thread 2
        res2 = supervisor.invoke("And if it were a reference book?", thread_id="thread-isolated-2")
        # Should ask for clarification rather than leaking 5 days from thread 1
        self.assertTrue(
            "specify" in res2["final_response"].lower()
            or "overdue" in res2["final_response"].lower()
            or res2["intent"] == "clarification_needed"
        )


if __name__ == "__main__":
    unittest.main()
