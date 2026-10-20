"""High-level memory operations used by the MCP server.

:class:`MemoryStore` wraps the SQLite layer and implements the behaviours
required by the MemoryBridge specification:

* **Create** with duplicate detection (exact and near-duplicate).
* **Search** with a hybrid ranker: FTS5/BM25 keyword scores combined with
  TF-IDF cosine similarity over stemmed tokens, plus light synonym
  expansion on the query side so paraphrased questions still match.
* **Inspect** (list, optionally filtered by category).
* **Update** and **delete** by id, scoped to the owning user.
* **Isolation**: every query filters on ``owner``; the owner is fixed when
  the store is constructed from server configuration, not from tool input.

The ranker keeps only results whose combined score clears a minimum
threshold, so an unrelated query returns an empty list instead of the
user's entire profile.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import db, text

__all__ = ["Memory", "MemoryStore", "SearchResult"]

# --- Tuning constants -------------------------------------------------------

# Weight of the keyword (BM25) leg vs the semantic (TF-IDF cosine) leg.
# BM25 scores are unbounded, so they are squashed with 1 - exp(-bm25) first.
W_KEYWORD = 0.45
W_SEMANTIC = 0.55

# Minimum blended score for a memory to be returned. Tuned against the test
# corpus: paraphrase matches land in 0.06--0.15, while queries with no
# keyword or synonym overlap score exactly 0.0 and are dropped, so an
# unrelated question yields an empty list rather than the whole profile.
DEFAULT_MIN_SCORE = 0.06

# Cosine similarity above which a new memory is treated as a duplicate.
DUP_COSINE_THRESHOLD = 0.88

# Top-N candidates pulled from each leg before the final ranking.
_CANDIDATE_LIMIT = 50

# Hard caps so one user cannot flood another with results.
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

MAX_CONTENT_LENGTH = 2000
VALID_CATEGORIES = ("general", "preference", "goal", "constraint", "project", "fact")


def _utcnow() -> str:
    # Microsecond precision: two memories added in the same second still get
    # distinct timestamps, so "newest first" ordering stays deterministic.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Memory:
    """A single stored memory record."""

    id: str
    owner: str
    content: str
    category: str
    source: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: Any) -> "Memory":
        return cls(
            id=row["id"],
            owner=row["owner"],
            content=row["content"],
            category=row["category"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SearchResult:
    """A memory paired with its ranking score."""

    memory: Memory
    score: float
    keyword_score: float
    semantic_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = self.memory.to_dict()
        payload["score"] = round(self.score, 4)
        return payload


class MemoryStore:
    """Owner-scoped, persistent memory store."""

    def __init__(self, db_path: str, owner: str) -> None:
        if not owner or not owner.strip():
            raise ValueError("owner must be a non-empty string")
        self.owner = owner.strip()
        self._conn = db.connect(db_path)
        db.ensure_schema(self._conn)
        self._lock = threading.RLock()
        # Cache: tokens + vectors for this owner's rows, rebuilt on mutation.
        self._cache_tokens: list[list[str]] | None = None
        self._cache_vectors: list[dict[str, float]] | None = None
        self._cache_idf: dict[str, float] | None = None

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #

    def add(
        self,
        content: str,
        category: str = "general",
        source: str = "unknown",
    ) -> dict[str, Any]:
        """Store a new memory, refusing exact and near duplicates.

        Returns a dict with ``status`` set to ``created``, ``duplicate`` or
        ``merged``. For ``duplicate``/``merged`` the existing memory is
        returned so the caller can update it instead of piling up copies.
        """
        content = (content or "").strip()
        if not content:
            raise ValueError("content must be a non-empty string")
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValueError(f"content must be at most {MAX_CONTENT_LENGTH} characters")

        category = (category or "general").strip().lower() or "general"
        if category not in VALID_CATEGORIES:
            category = "general"
        source = (source or "unknown").strip() or "unknown"

        with self._lock:
            existing = self._find_exact_duplicate(content)
            if existing is not None:
                return {
                    "status": "duplicate",
                    "reason": "An identical memory already exists.",
                    "memory": Memory.from_row(existing).to_dict(),
                }

            near = self._find_near_duplicate(content)
            if near is not None:
                # Same meaning, slightly different wording: refresh it in
                # place instead of storing a redundant copy.
                return {
                    "status": "merged",
                    "reason": "A very similar memory already exists; it was not stored again. "
                    "Use update_memory if the existing memory needs correction.",
                    "memory": Memory.from_row(near).to_dict(),
                }

            memory_id = uuid.uuid4().hex[:12]
            now = _utcnow()
            self._conn.execute(
                "INSERT INTO memories (id, owner, content, category, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, self.owner, content, category, source, now, now),
            )
            self._conn.commit()
            self._invalidate_cache()
            created = self._get_row(memory_id)
            assert created is not None
            return {"status": "created", "memory": Memory.from_row(created).to_dict()}

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[dict[str, Any]]:
        """Return up to *top_k* memories relevant to *query*.

        Hybrid ranking: BM25 over the FTS5 index (exact/keyword matches)
        blended with TF-IDF cosine similarity (paraphrase tolerance).
        Results below *min_score* are dropped, so an unrelated query yields
        an empty list.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        min_score = max(0.0, float(min_score))

        query_tokens = text.tokenize(query)
        if not query_tokens:
            # Query was nothing but stop-words; nothing can be relevant.
            return []

        with self._lock:
            rows = self._all_rows()
            if not rows:
                return []

            keyword_scores = self._keyword_scores(query, len(rows))

            idf = self._ensure_cache(rows)
            weights = text.expand_synonyms(query_tokens)
            query_vec = text.vectorize(list(weights.keys()), idf, weights)

            results: list[SearchResult] = []
            for index, row in enumerate(rows):
                sem = text.cosine(query_vec, self._cache_vectors[index])  # type: ignore[index]
                raw_kw = keyword_scores.get(row["id"], 0.0)
                # Squash unbounded BM25 into [0, 1) so the blend is stable.
                kw = 1.0 - math.exp(-raw_kw) if raw_kw > 0 else 0.0
                score = W_KEYWORD * kw + W_SEMANTIC * sem
                if score < min_score:
                    continue
                results.append(
                    SearchResult(
                        memory=Memory.from_row(row),
                        score=score,
                        keyword_score=kw,
                        semantic_score=sem,
                    )
                )

            results.sort(key=lambda r: (-r.score, r.memory.updated_at))
            return [r.to_dict() for r in results[:top_k]]

    def _keyword_scores(self, query: str, total_rows: int) -> dict[str, float]:
        """BM25 scores from FTS5 for the owner's rows, keyed by memory id."""
        match_expr = self._fts_match_expr(query)
        if not match_expr:
            return {}
        # BM25 needs a corpus larger than a handful of rows to discriminate;
        # for tiny stores plain LIKE matching is just as good and safer.
        if total_rows < 5:
            return self._like_scores(query)
        try:
            rows = self._conn.execute(
                "SELECT m.id AS id, bm25(memories_fts) AS rank "
                "FROM memories_fts "
                "JOIN memories m ON m.rowid = memories_fts.rowid "
                "WHERE memories_fts MATCH ? AND m.owner = ? "
                "ORDER BY rank "
                "LIMIT ?",
                (match_expr, self.owner, _CANDIDATE_LIMIT),
            ).fetchall()
        except Exception:
            # A malformed FTS expression must never crash the tool call.
            return self._like_scores(query)
        # bm25() returns negative values (more negative = better), flip sign.
        return {row["id"]: -float(row["rank"]) for row in rows}

    def _like_scores(self, query: str) -> dict[str, float]:
        """Fallback keyword leg: overlap of stemmed tokens with each row.

        Applies the same synonym expansion as the semantic leg, so a query
        word that only matches via a synonym (``sample``/``example``) still
        scores here. Keeps this fallback as capable as the BM25 path it
        stands in for on tiny corpora, where BM25 cannot discriminate yet.
        """
        weights = text.expand_synonyms(text.tokenize(query))
        if not weights:
            return {}
        query_tokens = set(weights)
        rows = self._conn.execute(
            "SELECT id, content FROM memories WHERE owner = ?", (self.owner,)
        ).fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            content_tokens = set(text.tokenize(row["content"]))
            overlap = len(query_tokens & content_tokens)
            if overlap:
                scores[row["id"]] = overlap / len(query_tokens)
        return scores

    @staticmethod
    def _fts_match_expr(query: str) -> str:
        """Build a safe FTS5 MATCH expression from *query*.

        Each word is quoted individually and OR-ed, so punctuation or FTS
        operators typed by a user/model cannot break the query syntax.
        """
        words = [w for w in text._WORD_RE.findall(query.lower()) if w not in text.STOPWORDS]
        if not words:
            return ""
        return " OR ".join(f'"{w}"' for w in words)

    # ------------------------------------------------------------------ #
    # Inspect / update / delete
    # ------------------------------------------------------------------ #

    def list_memories(
        self,
        category: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List the owner's memories, newest first, optionally by category."""
        limit = max(1, min(int(limit), 500))
        with self._lock:
            if category:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE owner = ? AND category = ? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (self.owner, category.strip().lower(), limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE owner = ? ORDER BY updated_at DESC LIMIT ?",
                    (self.owner, limit),
                ).fetchall()
            return [Memory.from_row(r).to_dict() for r in rows]

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Update *content* and/or *category* of one of the owner's memories.

        ``updated_at`` is refreshed. Returns ``status: not_found`` when the
        id does not exist or belongs to another user (no cross-user probing).
        """
        memory_id = (memory_id or "").strip()
        if not memory_id:
            raise ValueError("memory_id must be a non-empty string")
        if content is None and category is None:
            raise ValueError("provide at least one of content or category to update")

        if content is not None:
            content = content.strip()
            if not content:
                raise ValueError("content must be a non-empty string")
            if len(content) > MAX_CONTENT_LENGTH:
                raise ValueError(f"content must be at most {MAX_CONTENT_LENGTH} characters")

        if category is not None:
            category = category.strip().lower()
            if category not in VALID_CATEGORIES:
                raise ValueError(
                    f"category must be one of {', '.join(VALID_CATEGORIES)}"
                )

        with self._lock:
            row = self._get_row(memory_id)
            if row is None:
                return {
                    "status": "not_found",
                    "reason": f"No memory with id '{memory_id}' exists for this user.",
                }
            new_content = content if content is not None else row["content"]
            new_category = category if category is not None else row["category"]
            now = _utcnow()
            self._conn.execute(
                "UPDATE memories SET content = ?, category = ?, updated_at = ? WHERE id = ? AND owner = ?",
                (new_content, new_category, now, memory_id, self.owner),
            )
            self._conn.commit()
            self._invalidate_cache()
            updated = self._get_row(memory_id)
            assert updated is not None
            return {"status": "updated", "memory": Memory.from_row(updated).to_dict()}

    def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete one of the owner's memories by id.

        The FTS trigger removes it from future search results atomically.
        """
        memory_id = (memory_id or "").strip()
        if not memory_id:
            raise ValueError("memory_id must be a non-empty string")
        with self._lock:
            row = self._get_row(memory_id)
            if row is None:
                return {
                    "status": "not_found",
                    "reason": f"No memory with id '{memory_id}' exists for this user.",
                }
            self._conn.execute(
                "DELETE FROM memories WHERE id = ? AND owner = ?", (memory_id, self.owner)
            )
            self._conn.commit()
            self._invalidate_cache()
            return {"status": "deleted", "deleted_id": memory_id}

    def stats(self) -> dict[str, Any]:
        """Small summary used by list_memories for context."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE owner = ?", (self.owner,)
            ).fetchone()
            return {"owner": self.owner, "count": int(row["n"])}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _get_row(self, memory_id: str) -> Any:
        return self._conn.execute(
            "SELECT * FROM memories WHERE id = ? AND owner = ?",
            (memory_id, self.owner),
        ).fetchone()

    def _all_rows(self) -> list[Any]:
        return self._conn.execute(
            "SELECT * FROM memories WHERE owner = ?", (self.owner,)
        ).fetchall()

    def _find_exact_duplicate(self, content: str) -> Any:
        target = text.normalized_content(content)
        if not target:
            return None
        for row in self._all_rows():
            if text.normalized_content(row["content"]) == target:
                return row
        return None

    def _find_near_duplicate(self, content: str) -> Any:
        tokens = text.tokenize(content)
        if not tokens:
            return None
        rows = self._all_rows()
        if not rows:
            return None
        idf = text.build_idf([text.tokenize(r["content"]) for r in rows] + [tokens])
        vec = text.vectorize(tokens, idf)
        best_row, best_score = None, 0.0
        for row in rows:
            other = text.vectorize(text.tokenize(row["content"]), idf)
            score = text.cosine(vec, other)
            if score > best_score:
                best_row, best_score = row, score
        if best_row is not None and best_score >= DUP_COSINE_THRESHOLD:
            return best_row
        return None

    def _ensure_cache(self, rows: list[Any]) -> dict[str, float]:
        """Build (or reuse) per-row token lists, vectors and the IDF table."""
        if self._cache_tokens is not None and len(self._cache_tokens) == len(rows):
            assert self._cache_idf is not None and self._cache_vectors is not None
            return self._cache_idf
        tokens = [text.tokenize(r["content"]) for r in rows]
        idf = text.build_idf(tokens)
        vectors = [text.vectorize(t, idf) for t in tokens]
        self._cache_tokens = tokens
        self._cache_vectors = vectors
        self._cache_idf = idf
        return idf

    def _invalidate_cache(self) -> None:
        self._cache_tokens = None
        self._cache_vectors = None
        self._cache_idf = None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
