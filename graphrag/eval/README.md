# GraphRAG evaluation

`questions.jsonl` holds 40 hand-written questions, 10 per intent (`repurpose`, `explain_pair`,
`neighbours`, `free_query`). Each one has a gold Cypher query and the key column(s) to compare.
`graphrag/eval.py` runs every question through the full pipeline (route → link → template or
text2cypher → answer) against the full PrimeKG graph in Neo4j and compares result sets.

```bash
uv run --extra graph --extra llm python -m graphrag.eval          # replay recordings/ (no API calls)
uv run --extra graph --extra llm python -m graphrag.eval --live   # call Anthropic for missing ones
```

**Scoring.** A question is correct when, for every gold key column, some column of the
pipeline's rows holds exactly the same set of values. Column names and extra columns are
ignored, so `RETURN count(d) AS drug_count` matches a gold `n`. For `explain_pair` the gold is
the (drug id, disease id) pair, so it scores routing and entity linking; the DWPC paths
themselves are tested in Task 5.

## Result (Anthropic `claude-sonnet-5-5`, full graph)

| Metric | Value |
| --- | --- |
| Execution accuracy | **40/40 (100%)** (target ≥ 80%) |
| repurpose / explain_pair / neighbours / free_query | 10/10 each |
| Routing accuracy | 40/40 |
| Clarification prompts | 0 |
| text2cypher retries | 0 |
| Guard rejections | 0 |

Full per-question output: `results.json`. The LLM responses are in `recordings/` (keyed by
prompt hash), so the run above can be replayed offline.

**Caveats; read the 100% as "works on easy questions", not a benchmark.**
- I wrote the questions myself, knowing the schema, and they use canonical names
  ("Parkinson disease", "type 2 diabetes mellitus"). Misspelt or colloquial names
  ("parkinsons", "diabetes") are ambiguous in PrimeKG and get a clarification
  prompt instead of an answer, which this set doesn't measure.
- Gold sets were kept at ≤ 50 rows so the automatic `LIMIT 50` can't truncate them.
- Only 10 questions go through text2cypher and all are 1–2 hop. Multi-hop aggregations
  and questions about properties are untested.

## Adversarial requests

`eval.py` also sends five write/injection requests (delete all drugs, rename a drug, create a
node, `CALL dbms.security.listUsers()`, `LOAD CSV`) to text2cypher. In every case the model
itself refused and returned a harmless read query (or `WHERE false`), so the guard never had to
reject anything. The guard is tested directly on 13 malicious queries in `tests/test_guards.py`,
and the Neo4j client runs everything in a read-only transaction as a final backstop.
