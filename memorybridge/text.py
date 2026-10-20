"""Text utilities shared by storage and search.

This module is deliberately dependency-free. It provides:

* :func:`tokenize` -- lowercase word extraction with a small stop-word list.
* :func:`stem` -- a very small, predictable suffix stripper that unifies
  obvious variants such as ``learning`` / ``learn`` or ``trees`` / ``tree``.
* :func:`expand_synonyms` -- a curated, deterministic synonym expansion used
  only on the *query* side so that paraphrased questions still match stored
  memories (e.g. "study" <-> "learn", "plan" <-> "schedule").
* :func:`normalized_content` -- canonical form used for exact-duplicate
  detection.
* TF-IDF helpers (:func:`build_idf`, :func:`vectorize`, :func:`cosine`)
  used by the hybrid ranker in :mod:`memorybridge.store`.

Everything here is pure and deterministic, which keeps tests stable.
"""

from __future__ import annotations

import re
from collections import Counter

__all__ = [
    "STOPWORDS",
    "SYNONYMS",
    "build_idf",
    "cosine",
    "expand_synonyms",
    "normalized_content",
    "stem",
    "tokenize",
    "vectorize",
]

_WORD_RE = re.compile(r"[a-z0-9]+")

# Consonants that plausibly precede a silent "e" in verb stems (for the
# stemming heuristic below). Excludes n/f/h/w/x/y, which is why "opening"
# stays "open" and "walking" stays "walk".
_RESTORE_E_CONSONANTS = frozenset("bc dgklmprstvz")

# A small English stop-word list. Keeping it tiny avoids removing words that
# carry meaning in short memory sentences (e.g. "not", "no").
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did",
        "do", "does", "for", "from", "had", "has", "have", "he", "her", "him",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
        "no", "not", "of", "on", "or", "our", "out", "she", "so", "than",
        "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "those", "to", "too", "up", "us", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would",
        "you", "your",
    }
)

# Synonym groups keyed by *stemmed* word. Expansion happens only for queries,
# with a reduced weight, so it nudges ranking without overwhelming the actual
# query terms. Groups are symmetric where it makes sense.
_SYNONYM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Each group: anchor word followed by related words, most related first.
    # The map is made symmetric below, so "sample" also reaches "example".
    ("study", ("learn", "practice", "revision", "coursework", "tutoring", "revise")),
    ("plan", ("schedule", "roadmap", "routine", "timetable", "curriculum")),
    ("like", ("prefer", "preference", "favourite", "favorite")),
    ("available", ("availability", "free", "time")),
    ("daily", ("day", "everyday")),
    ("use", ("utilize", "using")),
    ("remember", ("memory", "memorize")),
    ("project", ("work", "endeavor")),
    ("job", ("work", "career", "employment")),
    ("work", ("job", "career", "office")),
    ("goal", ("objective", "aim")),
    ("constraint", ("limit", "limitation", "restriction")),
    ("deadline", ("due", "schedule", "timeline")),
    ("doctor", ("physician", "medical")),
    ("buy", ("purchase",)),
    ("quick", ("fast", "rapid")),
    ("language", ("programming", "lang")),
    ("example", ("sample", "illustration")),
    ("beginner", ("novice", "starter")),
    ("advanced", ("expert", "experienced")),
)


def _build_synonym_map() -> dict[str, tuple[str, ...]]:
    """Make every group symmetric: if A lists B, B lists A."""
    related: dict[str, set[str]] = {}
    for anchor, words in _SYNONYM_GROUPS:
        members = {anchor, *words}
        for member in members:
            related.setdefault(member, set()).update(members - {member})
    return {
        word: tuple(sorted(others, key=lambda w: (len(w), w)))
        for word, others in related.items()
    }


SYNONYMS: dict[str, tuple[str, ...]] = _build_synonym_map()

# Only the N most-related synonyms per token are used, keeping queries
# focused (and keeping the score's denominator small).
SYNONYM_TOP_N = 2

# Weight given to synonym tokens relative to literal query tokens.
SYNONYM_WEIGHT = 0.85


def stem(word: str) -> str:
    """Reduce *word* to a crude stem using a handful of safe rules.

    The rules are applied in order and never shrink a word below three
    characters, which keeps the transformation predictable:

    ``studies``/``studied`` -> ``study``, ``learning`` -> ``learn``,
    ``planning`` -> ``plan``, ``trees`` -> ``tree``.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        # "planning" -> "plann" -> "plan"
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            base = base[:-1]
        # Silent-e restoration: "hiking" -> "hike", "caring" -> "care".
        # Conditions: final letter is a consonant that can precede a silent
        # "e" (so "opening" -> "open" and "walking" -> "walk" are untouched),
        # preceded by a vowel, preceded by a consonant (so "waiting" ->
        # "wait", not "waite").
        elif (
            len(base) >= 3
            and base[-1] in _RESTORE_E_CONSONANTS
            and base[-2] in "aeiou"
            and base[-3] not in "aeiou"
        ):
            base += "e"
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            base = base[:-1]
        elif (
            len(base) >= 3
            and base[-1] in _RESTORE_E_CONSONANTS
            and base[-2] in "aeiou"
            and base[-3] not in "aeiou"
        ):
            base += "e"
        return base
    if word.endswith(("ches", "shes", "xes", "zes", "sses")) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, de-stop-word, and stem *text* into a token list."""
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        tokens.append(stem(raw))
    return tokens


def expand_synonyms(tokens: list[str]) -> dict[str, float]:
    """Return a token -> weight map, adding low-weight synonyms.

    Original tokens keep weight ``1.0``; the most-related synonyms get
    weight ``SYNONYM_WEIGHT`` so they influence ranking without
    overwhelming the literal query terms.
    """
    weights: dict[str, float] = {}
    for token in tokens:
        weights[token] = max(weights.get(token, 0.0), 1.0)
        for syn in SYNONYMS.get(token, ())[:SYNONYM_TOP_N]:
            syn_stem = stem(syn)
            if syn_stem != token:
                weights[syn_stem] = max(weights.get(syn_stem, 0.0), SYNONYM_WEIGHT)
    return weights

# Collapse whitespace and strip punctuation for exact-duplicate comparison.
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def normalized_content(text: str) -> str:
    """Canonical form of *text* used for exact-duplicate detection."""
    return _NON_WORD_RE.sub(" ", text.lower()).strip()


def build_idf(documents: list[list[str]]) -> dict[str, float]:
    """Smooth IDF over a list of tokenised documents.

    ``idf(t) = log((N + 1) / (df + 1)) + 1`` -- always positive, so a term
    present in every document still contributes a little.
    """
    import math

    total = len(documents)
    if total == 0:
        return {}
    df: Counter[str] = Counter()
    for doc in documents:
        df.update(set(doc))
    return {term: math.log((total + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def vectorize(
    tokens: list[str],
    idf: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build an L2-normalised TF-IDF vector for *tokens*.

    *weights* optionally scales individual tokens (used for synonym
    expansion). Unknown terms fall back to the mean IDF so out-of-vocabulary
    query words still count proportionally to their frequency.
    """
    import math

    counts = Counter(tokens)
    if not counts:
        return {}
    default_idf = (sum(idf.values()) / len(idf)) if idf else 1.0
    vec: dict[str, float] = {}
    for term, tf in counts.items():
        scale = 1.0 if weights is None else weights.get(term, 0.0)
        if scale == 0.0:
            continue
        vec[term] = tf * idf.get(term, default_idf) * scale
    norm = math.sqrt(sum(value * value for value in vec.values()))
    if norm == 0.0:
        return {}
    return {term: value / norm for term, value in vec.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosine similarity between two normalised vectors (0.0 -- 1.0)."""
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())
