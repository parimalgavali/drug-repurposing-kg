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
DISCLAIMER_DE = (
    "Forschungsdemo, keine medizinische Beratung: Graph-Verknüpfungen und Modellvorhersagen sind "
    "Hypothesen, kein Beleg dafür, dass ein Medikament sicher oder wirksam ist."
)
TREATMENT = re.compile(
    r"\b(treat|therap|drug|medication|cure|repurpos|indicat|candidate|prescri"
    r"|behandl|medikament|arznei|wirkstoff|kandidat|heil)",
    re.I,
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


def prompt(
    question: str, results: dict, lang: str = "en", names_de: dict[int, str] | None = None
) -> str:
    payload = json.dumps(results, ensure_ascii=False, default=str)
    text = f"Question: {question}\n\nQuery results (JSON):\n{payload}"
    if lang == "de":
        text += (
            "\n\nAnswer in German. Use these German names for the ids where given "
            "and keep every [id] citation unchanged; translate other names only if you are sure, "
            f"otherwise keep them in English:\n{json.dumps(names_de or {}, ensure_ascii=False)}"
        )
    return text


def answer(provider: Provider, question: str, intent: str, results: dict) -> str:
    text = provider.complete(SYSTEM, prompt(question, results), max_tokens=700).strip()
    if needs_disclaimer(question, intent):
        text += f"\n\n_{DISCLAIMER}_"
    return text


def stream_answer(
    provider: Provider,
    question: str,
    intent: str,
    results: dict,
    lang: str = "en",
    names_de: dict[int, str] | None = None,
) -> Iterator[str]:
    yield from provider.stream(SYSTEM, prompt(question, results, lang, names_de), max_tokens=700)
    if needs_disclaimer(question, intent):
        yield f"\n\n_{DISCLAIMER_DE if lang == 'de' else DISCLAIMER}_"


def cited_ids(text: str) -> set[int]:
    return {int(m) for m in re.findall(r"\[(\d+)\]", text)}
