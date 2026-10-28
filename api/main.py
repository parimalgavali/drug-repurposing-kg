"""RepurposeKG HTTP API: search, candidates, evidence paths and streamed GraphRAG answers.

uv run --extra graph --extra llm --extra api uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.settings import Settings
from explain.paths import explain
from graphrag.answer import MAX_ROWS, stream_answer
from graphrag.linker import link
from graphrag.pipeline import Plan, plan
from graphrag.router import Label
from graphrag.templates import KNOWN, REPURPOSE
from kg.client import KGClient, connect
from llm import Provider, get_provider

log = logging.getLogger(__name__)
NODE = "MATCH (n:{label} {{id: $id}}) RETURN n.id AS id, n.name AS name, labels(n)[0] AS label"


class Node(BaseModel):
    id: int
    name: str
    label: str


class SearchHit(Node):
    confidence: float


class Candidate(BaseModel):
    drug_id: int
    drug: str
    rank: int
    score: float
    ct_studies: int | None = None
    ct_max_phase: float | None = None


class KnownDrug(BaseModel):
    drug_id: int
    drug: str
    relation: str


class Candidates(BaseModel):
    disease: Node
    candidates: list[Candidate]
    known: list[KnownDrug]


class PathNode(BaseModel):
    id: int
    name: str
    label: str


class EvidencePath(BaseModel):
    nodes: list[PathNode]
    rels: list[str]
    pdp: float
    text: str


class Metapath(BaseModel):
    metapath: str
    dwpc: float
    count: int
    paths: list[EvidencePath]


class CyElement(BaseModel):
    data: dict[str, str]


class Cytoscape(BaseModel):
    nodes: list[CyElement]
    edges: list[CyElement]


class Explanation(BaseModel):
    drug: Node
    disease: Node
    paths_found: int
    truncated: bool
    metapaths: list[Metapath]
    cytoscape: Cytoscape


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def get_kg(request: Request) -> KGClient:
    return request.app.state.kg


def get_llm(request: Request) -> Provider | None:
    return request.app.state.provider


KG = Annotated[KGClient, Depends(get_kg)]
LLM = Annotated[Provider | None, Depends(get_llm)]
router = APIRouter(prefix="/api")


def node_or_404(kg: KGClient, label: str, node_id: int) -> Node:
    rows = kg.read(NODE.format(label=label), {"id": node_id})
    if not rows:
        raise HTTPException(404, f"no {label} with id {node_id}")
    return Node(**rows[0])


@router.get("/health")
def health(kg: KG, provider: LLM) -> JSONResponse:
    try:
        neo4j = kg.read("RETURN 1 AS ok", timeout=3)[0]["ok"] == 1
    except Exception:
        neo4j = False
    body = {"status": "ok" if neo4j else "degraded", "neo4j": neo4j}
    body["llm"] = provider.name if provider else None
    return JSONResponse(body, status_code=200 if neo4j else 503)


@router.get("/search")
def search(
    kg: KG,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    type: Label | None = None,
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
) -> list[SearchHit]:
    found = link(kg, q, (type,) if type else (), top=limit)
    return [
        SearchHit(id=c.id, name=c.name, label=c.label, confidence=c.confidence)
        for c in found.candidates
    ]


@router.get("/diseases/{disease_id}/candidates")
def candidates(
    kg: KG, disease_id: int, limit: Annotated[int, Query(ge=1, le=50)] = 20
) -> Candidates:
    disease = node_or_404(kg, "Disease", disease_id)
    params = {"disease": disease_id, "k": limit}
    return Candidates(
        disease=disease,
        candidates=kg.read(REPURPOSE, params),
        known=kg.read(KNOWN, params),
    )


@router.get("/pairs/{drug_id}/{disease_id}/explain")
def explain_pair(kg: KG, drug_id: int, disease_id: int) -> Explanation:
    drug = node_or_404(kg, "Drug", drug_id)
    disease = node_or_404(kg, "Disease", disease_id)
    return Explanation(**{**explain(kg, drug_id, disease_id), "drug": drug, "disease": disease})


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def subgraph(p: Plan) -> dict:
    """Cytoscape JSON for the answer: evidence paths, or linked entities plus the returned rows."""
    if p.evidence:
        return p.evidence["cytoscape"]
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    def add(node_id, name, label) -> str:
        key = str(node_id)
        nodes[key] = {"data": {"id": key, "label": str(name), "type": str(label)}}
        return key

    hubs = [lk for lk in p.links if "id" in lk]
    for h in hubs:
        add(h["id"], h["name"], h["label"])
    hub = str(hubs[0]["id"]) if len(hubs) == 1 else None
    for row in p.rows[:MAX_ROWS]:
        if "drug_id" in row:
            key = add(row["drug_id"], row.get("drug"), "Drug")
            rel = row.get("relation", "PREDICTED_INDICATION")
        elif "id" in row and "name" in row:
            key = add(row["id"], row["name"], row.get("label", "Node"))
            rel = row.get("relation", "RELATED")
        else:
            continue
        if hub and key != hub:
            edge = f"{hub}-{rel}-{key}"
            edges[edge] = {"data": {"id": edge, "source": hub, "target": key, "label": str(rel)}}
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def ask_events(question: str, provider: Provider | None, kg: KGClient) -> Iterator[str]:
    """SSE: entities → cypher → rows → token… → subgraph → done; `error` ends the stream early."""
    if provider is None:
        yield sse("error", {"message": "No LLM provider is configured."})
        return
    try:
        p = plan(question, provider, kg)
        yield sse("entities", p.links)
        if p.clarification:
            yield sse("token", {"text": p.clarification})
            yield sse("done", {"intent": p.intent, "clarification": True})
            return
        yield sse("cypher", {"queries": p.cypher, "attempts": p.attempts})
        if p.error:
            yield sse(
                "error",
                {"message": "Could not answer with a safe read-only query.", "detail": p.error},
            )
            return
        yield sse("rows", {"rows": p.rows[:MAX_ROWS], "total": len(p.rows)})
        for token in stream_answer(provider, question, p.intent, p.results):
            yield sse("token", {"text": token})
        yield sse("subgraph", subgraph(p))
        yield sse("done", {"intent": p.intent, "clarification": False})
    except Exception as e:
        log.exception("ask failed")
        yield sse(
            "error", {"message": "The question could not be answered.", "detail": type(e).__name__}
        )


@router.post(
    "/ask",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE event stream"}},
)
def ask(body: AskRequest, kg: KG, provider: LLM) -> StreamingResponse:
    return StreamingResponse(
        ask_events(body.question, provider, kg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(
    kg: KGClient | None = None, provider: Provider | None = None, settings: Settings | None = None
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.kg = kg or KGClient(
            connect(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password),
            settings.neo4j_database,
        )
        app.state.provider = provider
        if provider is None:
            try:
                app.state.provider = get_provider(settings.llm_provider)
            except Exception as e:  # e.g. no API key: everything but /ask still works
                log.warning("LLM provider %r unavailable: %s", settings.llm_provider, e)
        yield
        if kg is None:
            app.state.kg.close()

    app = FastAPI(
        title="RepurposeKG API",
        version="0.1.0",
        description="Research demo, not medical advice.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"]
    )
    app.include_router(router)
    return app


app = create_app()
