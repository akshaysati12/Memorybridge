"""The six scenarios the assignment requires to be demonstrated.

1. Save       -- a memory stored through one assistant.
2. Reuse      -- the same memory retrieved through a second assistant.
3. Paraphrase -- a memory retrieved with differently-worded query.
4. Update     -- an outdated fact corrected.
5. Delete     -- a memory deleted and gone from search.
6. Isolation  -- one user cannot access another user's memories.

Each test walks the *full* lifecycle the way the demo video would.
"""

from __future__ import annotations

from memorybridge.store import MemoryStore


class TestScenarioSaveAndReuse:
    def test_save_via_assistant_a_reuse_via_assistant_b(self, tmp_path):
        db_path = str(tmp_path / "demo.db")

        # --- Assistant A saves, in three separate memories (per spec). ---
        assistant_a = MemoryStore(db_path, owner="alice")
        assistant_a.add("I am learning machine learning", category="goal", source="assistant-a")
        assistant_a.add("I prefer Python examples", category="preference", source="assistant-a")
        assistant_a.add("I can study for one hour each day", category="constraint", source="assistant-a")
        assistant_a.close()  # assistant A disconnects

        # --- Assistant B connects to the same store, asks a related question. ---
        assistant_b = MemoryStore(db_path, owner="alice")
        try:
            results = assistant_b.search(
                "study plan for learning decision trees with code samples"
            )
            contents = " ".join(r["content"].lower() for r in results)
            assert "machine learning" in contents
            assert "python" in contents
            assert "one hour" in contents
        finally:
            assistant_b.close()


class TestScenarioParaphrasedSearch:
    def test_differently_worded_query_finds_memory(self, seeded_store):
        # Stored: "I prefer Python examples"
        results = seeded_store.search("which programming language does he want sample code in?")
        assert any("Python" in r["content"] for r in results)

    def test_case_and_word_order_variants(self, seeded_store):
        # Stored: "I can study for one hour each day"
        results = seeded_store.search("daily, how long is he free to revise?")
        assert any("one hour" in r["content"] for r in results)


class TestScenarioUpdate:
    def test_outdated_fact_is_replaced_not_duplicated(self, seeded_store):
        target = seeded_store.list_memories(category="constraint")[0]
        assert "one hour" in target["content"]

        result = seeded_store.update(target["id"], content="I can study for two hours each day")
        assert result["status"] == "updated"

        # The current fact is what search returns; the stale one is gone.
        results = seeded_store.search("how long can he study each day?")
        contents = [r["content"] for r in results]
        assert any("two hours" in c for c in contents)
        assert not any("one hour" in c for c in contents)
        assert seeded_store.stats()["count"] == 5  # replaced, not added


class TestScenarioDelete:
    def test_deleted_memory_never_returned(self, seeded_store):
        target = seeded_store.list_memories(category="goal")[0]
        deleted_content = target["content"]
        assert seeded_store.delete(target["id"])["status"] == "deleted"

        # The deleted memory must never surface again, by content or id.
        contents = [m["content"] for m in seeded_store.list_memories()]
        assert deleted_content not in contents
        for result in seeded_store.search("machine learning"):
            assert result["id"] != target["id"]
            assert "machine learning" not in result["content"].lower()
        # Other, still-relevant memories may legitimately match the query.
        assert seeded_store.stats()["count"] == 4


class TestScenarioIsolation:
    def test_user_b_cannot_read_update_or_delete_user_a(self, tmp_path):
        db_path = str(tmp_path / "iso.db")
        alice = MemoryStore(db_path, owner="alice")
        secret = alice.add("My therapy appointment is Thursdays at 3pm", category="fact")["memory"]

        bob = MemoryStore(db_path, owner="bob")
        try:
            # Cannot read.
            assert bob.search("therapy appointment") == []
            assert all("Thursdays" not in m["content"] for m in bob.list_memories())
            # Cannot update.
            assert bob.update(secret["id"], content="hijacked")["status"] == "not_found"
            # Cannot delete.
            assert bob.delete(secret["id"])["status"] == "not_found"
            # Alice's memory is intact.
            assert alice.stats()["count"] == 1
        finally:
            bob.close()
            alice.close()
