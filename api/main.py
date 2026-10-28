"""RepurposeKG HTTP API: search, candidates, evidence paths and streamed GraphRAG answers.

uv run --extra graph --extra llm --extra api uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
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
from semweb.export import export as export_rdf
from semweb.store import RdfStore, SparqlError
from validation.ctgov import CTGov, search_url

log = logging.getLogger(__name__)
NODE = (
    "MATCH (n:{label} {{id: $id}}) "
    "RETURN n.id AS id, n.name AS name, n.name_de AS name_de, labels(n)[0] AS label"
)
GERMAN = """
UNWIND $ids AS id
OPTIONAL MATCH (d:Drug {id: id})
OPTIONAL MATCH (s:Disease {id: id})
WITH id, coalesce(d.name_de, s.name_de) AS name_de
WHERE name_de IS NOT NULL
RETURN id, name_de
"""
Lang = Literal["en", "de"]


class Node(BaseModel):
    id: int
    name: str
    name_de: str | None = None
    label: str


class SearchHit(Node):
    confidence: float


class Candidate(BaseModel):
    drug_id: int
    drug: str
    drug_de: str | None = None
    rank: int
    score: float
    ct_studies: int | None = None
    ct_max_phase: float | None = None


class KnownDrug(BaseModel):
    drug_id: int
    drug: str
    drug_de: str | None = None
    relation: str


class Candidates(BaseModel):
    disease: Node
    candidates: list[Candidate]
    known: list[KnownDrug]


class Trials(BaseModel):
    """ClinicalTrials.gov studies matching the disease and drug names (registered, not proof)."""

    studies: int
    max_phase: float | None
    statuses: dict[str, int]
    url: str
    cached: bool


class PathNode(BaseModel):
    id: int
    name: str
    name_de: str | None = None
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
    lang: Lang = "en"


def get_kg(request: Request) -> KGClient:
    return request.app.state.kg


def get_llm(request: Request) -> Provider | None:
    return request.app.state.provider


def get_ctgov(request: Request) -> CTGov:
    return request.app.state.ctgov


KG = Annotated[KGClient, Depends(get_kg)]
CT = Annotated[CTGov, Depends(get_ctgov)]
LLM = Annotated[Provider | None, Depends(get_llm)]
router = APIRouter(prefix="/api")


def german_names(kg: KGClient, ids) -> dict[int, str]:
    """German Wikidata names (`name_de`) for the drug/disease ids that have one."""
    ids = sorted({int(i) for i in ids if str(i).lstrip("-").isdigit()})
    if not ids:
        return {}
    return {r["id"]: r["name_de"] for r in kg.read(GERMAN, {"ids": ids})}


def with_german_labels(cy: dict, names: dict[int, str]) -> dict:
    """Adds `label_de` to Cytoscape nodes that have a German name."""
    for n in cy["nodes"]:
        name = names.get(int(n["data"]["id"])) if n["data"]["id"].isdigit() else None
        if name:
            n["data"]["label_de"] = name
    return cy


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
    de = german_names(kg, [c.id for c in found.candidates])
    return [
        SearchHit(
            id=c.id, name=c.name, name_de=de.get(c.id), label=c.label, confidence=c.confidence
        )
        for c in found.candidates
    ]


@router.get("/diseases/{disease_id}/candidates")
def candidates(
    kg: KG, ct: CT, disease_id: int, limit: Annotated[int, Query(ge=1, le=50)] = 20
) -> Candidates:
    disease = node_or_404(kg, "Disease", disease_id)
    params = {"disease": disease_id, "k": limit}
    preds, known = kg.read(REPURPOSE, params), kg.read(KNOWN, params)
    de = german_names(kg, [r["drug_id"] for r in preds + known])
    preds = [{**r, "drug_de": de.get(r["drug_id"])} for r in preds]
    for r in preds:  # counts stored by `make validate` win; else what /trials already looked up
        if r.get("ct_studies") is None and (s := ct.cached(disease.name, r["drug"])):
            r["ct_studies"], r["ct_max_phase"] = s.studies, s.max_phase
    return Candidates(
        disease=disease,
        candidates=preds,
        known=[{**r, "drug_de": de.get(r["drug_id"])} for r in known],
    )


@router.get("/pairs/{drug_id}/{disease_id}/explain")
def explain_pair(kg: KG, drug_id: int, disease_id: int) -> Explanation:
    drug = node_or_404(kg, "Drug", drug_id)
    disease = node_or_404(kg, "Disease", disease_id)
    found = explain(kg, drug_id, disease_id)
    de = german_names(kg, [n["data"]["id"] for n in found["cytoscape"]["nodes"]])
    for m in found["metapaths"]:
        for path in m["paths"]:
            path["nodes"] = [{**n, "name_de": de.get(n["id"])} for n in path["nodes"]]
    found["cytoscape"] = with_german_labels(found["cytoscape"], de)
    return Explanation(**{**found, "drug": drug, "disease": disease})


@router.get("/pairs/{drug_id}/{disease_id}/trials")
def pair_trials(kg: KG, ct: CT, drug_id: int, disease_id: int) -> Trials:
    drug = node_or_404(kg, "Drug", drug_id)
    disease = node_or_404(kg, "Disease", disease_id)
    cached = ct.cached(disease.name, drug.name) is not None
    try:
        s = ct.lookup(disease.name, drug.name)
    except (httpx.HTTPError, RuntimeError) as e:
        log.warning("CT.gov lookup failed for %s / %s: %s", disease.name, drug.name, e)
        raise HTTPException(502, "ClinicalTrials.gov lookup failed") from e
    url = search_url(disease.name, drug.name)
    return Trials(
        studies=s.studies, max_phase=s.max_phase, statuses=s.statuses, url=url, cached=cached
    )


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


def ask_events(
    question: str, provider: Provider | None, kg: KGClient, lang: Lang = "en"
) -> Iterator[str]:
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
        graph = subgraph(p)
        de = german_names(kg, [n["data"]["id"] for n in graph["nodes"]]) if lang == "de" else {}
        for token in stream_answer(provider, question, p.intent, p.results, lang, de):
            yield sse("token", {"text": token})
        yield sse("subgraph", with_german_labels(graph, de))
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
        ask_events(body.question, provider, kg, body.lang),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SparqlQuery(BaseModel):
    query: str = Field(min_length=1, max_length=5000)


def get_rdf(request: Request) -> RdfStore | None:
    return getattr(request.app.state, "rdf", None)


RDF = Annotated[RdfStore | None, Depends(get_rdf)]


def run_sparql(store: RdfStore | None, query: str) -> dict:
    if store is None:
        raise HTTPException(503, "no RDF export loaded; run `make rdf`")
    try:
        return store.query(query)
    except SparqlError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/sparql")
def sparql_get(store: RDF, query: Annotated[str, Query(min_length=1, max_length=5000)]) -> dict:
    """Read-only SPARQL (SELECT/ASK) over the exported RDF subgraph."""
    return run_sparql(store, query)


@router.post("/sparql")
def sparql_post(store: RDF, body: SparqlQuery) -> dict:
    return run_sparql(store, body.query)


@router.get("/diseases/{disease_id}/rdf", response_class=Response)
def disease_rdf(kg: KG, disease_id: int, k: Annotated[int, Query(ge=1, le=50)] = 20) -> Response:
    """Turtle export of the disease-centred subgraph (see docs/SEMANTIC_WEB.md)."""
    node_or_404(kg, "Disease", disease_id)
    turtle = export_rdf(kg, disease_id, k).serialize(format="turtle")
    return Response(turtle, media_type="text/turtle")


def create_app(
    kg: KGClient | None = None,
    provider: Provider | None = None,
    settings: Settings | None = None,
    rdf: RdfStore | None = None,
    ctgov: CTGov | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.kg = kg or KGClient(
            connect(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password),
            settings.neo4j_database,
        )
        app.state.provider = provider
        app.state.ctgov = ctgov or CTGov(cache=Path(settings.ctgov_cache), retries=2)
        rdf_path = Path(settings.rdf_path)
        app.state.rdf = rdf or (RdfStore.from_path(rdf_path) if rdf_path.exists() else None)
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
