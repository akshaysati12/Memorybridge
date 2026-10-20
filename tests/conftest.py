"""Shared fixtures for MemoryBridge tests."""

from __future__ import annotations

import pytest

from memorybridge.store import MemoryStore


@pytest.fixture()
def store(tmp_path):
    """An owner-scoped store backed by a temp file (persistence-capable)."""
    db_path = str(tmp_path / "test.db")
    s = MemoryStore(db_path, owner="alice")
    yield s
    s.close()


@pytest.fixture()
def other_store(tmp_path):
    """A second user sharing the same database file (isolation tests)."""
    db_path = str(tmp_path / "test.db")
    s = MemoryStore(db_path, owner="bob")
    yield s
    s.close()


SEED = [
    ("I am learning machine learning", "goal", "assistant-a"),
    ("I prefer Python examples", "preference", "assistant-a"),
    ("I can study for one hour each day", "constraint", "assistant-a"),
    ("My thesis deadline is 15 March", "project", "assistant-b"),
    ("I work as a data analyst at a bank", "fact", "assistant-b"),
]


@pytest.fixture()
def seeded_store(store):
    """A store with five synthetic memories for two fictional users."""
    for content, category, source in SEED:
        result = store.add(content, category=category, source=source)
        assert result["status"] == "created", result
    return store
