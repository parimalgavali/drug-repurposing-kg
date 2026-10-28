"""Intent routing with structured output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from graphrag.schema import LABELS
from llm import Provider

Label = Literal[LABELS]  # type: ignore[valid-type]


class Route(BaseModel):
    intent: Literal["repurpose", "explain_pair", "neighbours", "free_query"]
    drug: str | None = Field(None, description="Drug mention, verbatim from the question")
    disease: str | None = Field(None, description="Disease mention, verbatim from the question")
    entity: str | None = Field(None, description="For neighbours: the entity mention")
    entity_label: Label | None = Field(None, description="For neighbours: the entity's label")
    relation: str | None = Field(
        None, description="For neighbours: relationship type to follow, e.g. TARGET"
    )
    target_label: Label | None = Field(None, description="For neighbours: label of the neighbours")


SYSTEM = """Classify a question about a biomedical knowledge graph and extract its slots.

Intents:
- repurpose: which (new/candidate/predicted) drugs could treat a disease. Slots: disease.
- explain_pair: why or how a specific drug might relate to or treat a specific disease.
  Slots: drug, disease.
- neighbours: what is directly connected to one entity, e.g. a drug's targets, a gene's
  diseases, a disease's symptoms or approved drugs. Slots: entity, entity_label, and
  relation/target_label when the question names them.
- free_query: anything else (counts, multi-hop filters, comparisons, aggregations).

Copy mentions verbatim; do not normalise or expand them.
Relationship types for neighbours: INDICATION, CONTRAINDICATION, OFF_LABEL_USE, TARGET, ENZYME,
TRANSPORTER, CARRIER, DRUG_EFFECT, DISEASE_PROTEIN, DISEASE_PHENOTYPE_POSITIVE,
PROTEIN_PROTEIN, PATHWAY_PROTEIN, DISEASE_DISEASE."""


def route(provider: Provider, question: str) -> Route:
    return provider.structured(SYSTEM, f"Question: {question}", Route)
