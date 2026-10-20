# Test results

Run on Windows with Python 3.13, `mcp` 2.3.0, pytest 8.x.

```
60 passed in 9.11s
```

Command: `pytest` (from the repo root).

## Synthetic data

Test fixtures (`tests/conftest.py`) build a small in-memory corpus. It is
synthetic but shaped like real personal context — a mix of goals,
preferences, constraints, facts and project notes:

| content | category | source |
| --- | --- | --- |
| I am learning machine learning | goal | assistant-a |
| I prefer Python examples | preference | assistant-a |
| I can study for one hour each day | constraint | assistant-a |
| My thesis deadline is 15 March | project | assistant-b |
| I work as a data analyst at a bank | fact | assistant-b |

Two owner identities (`alice`, `bob`) share one database file where isolation
needs to be proven.

## The six required scenarios

Each maps to a dedicated test in `tests/test_scenarios.py`.

| # | Scenario | Test | Result |
| --- | --- | --- | --- |
| 1 | Save via one assistant | `TestScenarioSaveAndReuse` | pass |
| 2 | Reuse via a second assistant | `TestScenarioSaveAndReuse` | pass |
| 3 | Paraphrased search | `TestScenarioParaphrasedSearch` | pass |
| 4 | Update an outdated fact | `TestScenarioUpdate` | pass |
| 5 | Delete, gone from search | `TestScenarioDelete` | pass |
| 6 | Cross-user isolation | `TestScenarioIsolation` | pass |

Scenario 2 proves portability: assistant A saves and disconnects, then
assistant B (a separate process against the same database and owner) retrieves
the same memories without them being re-entered.

Scenario 3 covers rewording, not just keyword hits. A stored memory is found
by queries that share no exact wording with it:

* stored `I prefer Python examples` ← query `which programming language does
  he want sample code in?` (via the `language`/`example` synonym groups)
* stored `I can study for one hour each day` ← query `daily, how long is he
  free to revise?` (via `daily`/`revise` → `study`)

## Coverage beyond the six scenarios

The remaining 54 tests (`tests/test_store.py`, `tests/test_server.py`,
`tests/test_e2e.py`) cover the rest of the spec and the failure modes a demo
won't show on its own:

* **Duplicate handling** — identical content, punctuation/case variants, and
  near-duplicates (high TF-IDF cosine) are skipped or merged, never stored
  twice.
* **Persistence** — a store is closed and reopened against the same file;
  memories and deletions both survive.
* **Isolation** — cross-user `update`/`delete` report `not_found` (no probing
  by id), and the owner identity never appears in tool output.
* **Irrelevant queries return empty** — e.g. `zebra safari photography` over
  the corpus above yields `[]` rather than the whole profile. This is what
  keeps `min_score` honest.
* **Injection safety** — a memory containing `ignore all previous
  instructions` is stored and returned as inert data, and does not disrupt
  update/delete.
* **Robustness** — FTS5 operator characters (`"`, `*`, `(`) in a query do not
  crash search; Unicode content round-trips; empty and oversized content are
  rejected with an error rather than an exception leaking to the client.
* **Real MCP handshake** — `test_e2e.py` spawns the server over stdio and
  drives the full lifecycle through the wire protocol, including isolation
  and persistence across a restart.

## Tuning notes

`min_score` defaults to `0.06`, and the value was chosen against this corpus
rather than guessed. Paraphrase matches scored `0.06`–`0.15` while unrelated
queries scored exactly `0.0`, so `0.06` admits reworded matches while still
returning an empty list when nothing is relevant. The relevant-vs-irrelevant
gap is asserted directly (`test_irrelevant_query_returns_empty`), so a future
change to the ranking that erodes it will fail the suite rather than silently
degrade search quality.
