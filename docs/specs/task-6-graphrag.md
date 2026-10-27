# Task 6: GraphRAG

**Goal:** ask the knowledge graph questions in English and get answers that cite the Cypher and paths behind them.

**Deliverables**
- `llm/`: `Provider` protocol (`complete`, `stream`, `structured`), `AnthropicProvider`, `OllamaProvider`, `RecordedProvider` (replays JSON fixtures keyed by prompt hash; `--record` mode writes them).
- `graphrag/linker.py`: mentions → typed node candidates via the Neo4j full-text index, re-ranked with rapidfuzz; returns ids + confidence; asks for clarification when ambiguous.
- `graphrag/router.py`: structured-output intent `repurpose | explain_pair | neighbours | free_query` + slots.
- `graphrag/templates.py`: parametrised Cypher for the first three intents (re-uses task-5 `explain`).
- `graphrag/text2cypher.py`: schema-in-prompt generation with guards: parse labels/relationship types and reject anything not in the schema, reject write clauses (`CREATE|MERGE|DELETE|SET|REMOVE|CALL dbms|LOAD CSV`), append `LIMIT 50` if missing, `EXPLAIN` first, run in a read-only transaction with a 5 s timeout, and retry once on an error message.
- `graphrag/answer.py`: synthesis prompt that may only use returned rows; cites node ids; adds the not-medical-advice line for treatment questions.
- `graphrag/eval/questions.jsonl`: 40 hand-written questions (10 per intent) with gold Cypher; `eval.py` reports execution accuracy (result-set equality) and guard rejections.

**Tests:** guards reject 10 malicious or invalid queries; linker on the fixture; full pipeline offline with `RecordedProvider`.

**Done when:** ≥ 80% execution accuracy on the gold set with Anthropic (numbers recorded in `graphrag/eval/README.md`); CI runs offline.

**Secrets:** `ANTHROPIC_API_KEY` to record fixtures and measure the eval.
