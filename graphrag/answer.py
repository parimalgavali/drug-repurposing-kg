"""Answer synthesis restricted to the rows the graph returned."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from llm import Provider

DISCLAIMER = (
    "Research demo, not medical advice: graph links and model predictions are hypotheses, "
    "not evidence that a drug is safe or effective."
)
TREATMENT = re.compile(
    r"\b(treat|therap|drug|medication|cure|repurpos|indicat|candidate|prescri)", re.I
)
MAX_ROWS = 50

SYSTEM = """You answer questions about a biomedical knowledge graph (PrimeKG) using ONLY the
query results provided. Rules:
- Use only facts present in the results; never add outside medical knowledge.
- Cite every entity you mention with its node id in square brackets, e.g. Salbutamol [14254].
- If the results are empty or do not answer the question, say so plainly.
- Predicted links (PREDICTED_INDICATION, rank/score) are model hypotheses; label them as such.
- Be concise: a short paragraph or a short list."""


def needs_disclaimer(question: str, intent: str) -> bool:
    return intent in ("repurpose", "explain_pair") or bool(TREATMENT.search(question))


def prompt(question: str, results: dict) -> str:
    payload = json.dumps(results, ensure_ascii=False, default=str)
    return f"Question: {question}\n\nQuery results (JSON):\n{payload}"


def answer(provider: Provider, question: str, intent: str, results: dict) -> str:
    text = provider.complete(SYSTEM, prompt(question, results), max_tokens=700).strip()
    if needs_disclaimer(question, intent):
        text += f"\n\n_{DISCLAIMER}_"
    return text


def stream_answer(provider: Provider, question: str, intent: str, results: dict) -> Iterator[str]:
    yield from provider.stream(SYSTEM, prompt(question, results), max_tokens=700)
    if needs_disclaimer(question, intent):
        yield f"\n\n_{DISCLAIMER}_"


def cited_ids(text: str) -> set[int]:
    return {int(m) for m in re.findall(r"\[(\d+)\]", text)}
