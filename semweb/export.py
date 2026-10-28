"""Export a disease-centred subgraph from Neo4j to RDF (Turtle) with standard IRIs.

    make rdf DISEASE=33675   # writes data/rdf/export.ttl

Entities use their source ontology's IRIs (MONDO/HPO OBO PURLs, DrugBank, NCBI Gene via
identifiers.org); PrimeKG's grouped MONDO diseases get a local IRI linked to each member
via `rkg:groups`. Model predictions are n-ary `rkg:Prediction` nodes carrying score and rank.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from kg.client import KGClient, connect

RKG = Namespace("https://w3id.org/repurposekg/vocab#")
NODE = Namespace("https://w3id.org/repurposekg/node/")
PRED = Namespace("https://w3id.org/repurposekg/prediction/")
OBO = "http://purl.obolibrary.org/obo/"

CLASSES = {
    "Drug": RKG.Drug,
    "Disease": RKG.Disease,
    "Gene": RKG.Gene,
    "Phenotype": RKG.Phenotype,
}
PREDICATES = {
    "INDICATION": RKG.indication,
    "OFF_LABEL_USE": RKG.offLabelUse,
    "TARGET": RKG.target,
    "DISEASE_PROTEIN": RKG.diseaseProtein,
    "DISEASE_PHENOTYPE_POSITIVE": RKG.phenotype,
}

PREDICTIONS = """
MATCH (d:Drug)-[p:PREDICTED_INDICATION]->(:Disease {id: $disease})
RETURN d.id AS drug, p.score AS score, p.rank AS rank, p.model_version AS model_version,
       p.ct_studies AS ct_studies
ORDER BY p.rank LIMIT $k
"""
KNOWN = """
MATCH (d:Drug)-[:INDICATION|OFF_LABEL_USE]-(:Disease {id: $disease})
RETURN DISTINCT d.id AS drug
"""
EDGES = """
MATCH (a)-[r:INDICATION|OFF_LABEL_USE|TARGET|DISEASE_PROTEIN|DISEASE_PHENOTYPE_POSITIVE]-(b)
WHERE (a:Drug AND a.id IN $drugs AND (b:Gene OR (b:Disease AND b.id = $disease)))
   OR (a:Disease AND a.id = $disease AND (b:Gene OR b:Phenotype))
RETURN DISTINCT a.id AS src, type(r) AS rel, b.id AS dst
"""
NODES = """
MATCH (n) WHERE n.id IN $ids
RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, n.source AS source,
       n.source_id AS source_id
"""


def iri(source: str, source_id: str, served_id: int) -> URIRef:
    if source == "MONDO":
        return URIRef(f"{OBO}MONDO_{int(source_id):07d}")
    if source == "HPO":
        return URIRef(f"{OBO}HP_{int(source_id):07d}")
    if source == "DrugBank":
        return URIRef(f"https://go.drugbank.com/drugs/{source_id}")
    if source == "NCBI":
        return URIRef(f"https://identifiers.org/ncbigene:{source_id}")
    return NODE[str(served_id)]


def to_graph(
    disease: int, nodes: Iterable[dict], edges: Iterable[dict], predictions: Iterable[dict]
) -> Graph:
    g = Graph()
    g.bind("rkg", RKG)
    g.bind("node", NODE)
    g.bind("pred", PRED)
    g.bind("obo", OBO)
    g.bind("drugbank", "https://go.drugbank.com/drugs/")
    g.bind("ncbigene", "https://identifiers.org/ncbigene:")
    ids: dict[int, URIRef] = {}
    for n in nodes:
        if n["label"] not in CLASSES:
            continue
        ref = ids[n["id"]] = iri(n["source"], n["source_id"], n["id"])
        g.add((ref, RDF.type, CLASSES[n["label"]]))
        g.add((ref, RDFS.label, Literal(n["name"], datatype=XSD.string)))
        g.add((ref, RKG.servedId, Literal(n["id"], datatype=XSD.integer)))
        g.add((ref, RKG.source, Literal(n["source"], datatype=XSD.string)))
        if n["source"] == "MONDO_grouped":
            for member in str(n["source_id"]).split("_"):
                g.add((ref, RKG.groups, URIRef(f"{OBO}MONDO_{int(member):07d}")))
    for e in edges:
        if e["src"] in ids and e["dst"] in ids and e["rel"] in PREDICATES:
            g.add((ids[e["src"]], PREDICATES[e["rel"]], ids[e["dst"]]))
    for p in predictions:
        ref = PRED[f"{p['drug']}-{disease}"]
        g.add((ref, RDF.type, RKG.Prediction))
        g.add((ref, RKG.drug, ids[p["drug"]]))
        g.add((ref, RKG.disease, ids[disease]))
        g.add((ref, RKG.score, Literal(float(p["score"]), datatype=XSD.double)))
        g.add((ref, RKG.rank, Literal(int(p["rank"]), datatype=XSD.integer)))
        g.add((ref, RKG.modelVersion, Literal(str(p["model_version"]), datatype=XSD.string)))
        if p.get("ct_studies") is not None:
            g.add((ref, RKG.trialCount, Literal(int(p["ct_studies"]), datatype=XSD.integer)))
    return g


def export(kg: KGClient, disease: int, k: int = 20) -> Graph:
    """Disease + top-k predicted and known drugs, their targets, its genes and phenotypes."""
    params = {"disease": disease, "k": k}
    predictions = kg.read(PREDICTIONS, params)
    drugs = sorted({p["drug"] for p in predictions} | {r["drug"] for r in kg.read(KNOWN, params)})
    edges = kg.read(EDGES, {"disease": disease, "drugs": drugs})
    ids = {disease, *drugs, *(e["dst"] for e in edges)}
    nodes = kg.read(NODES, {"ids": sorted(ids)})
    return to_graph(disease, nodes, edges, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--disease", type=int, action="append", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data/rdf/export.ttl"))
    args = parser.parse_args()
    kg = KGClient(connect())
    try:
        g = Graph()
        for d in args.disease:
            g += export(kg, d, args.k)
    finally:
        kg.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(args.out, format="turtle")
    print(f"wrote {len(g)} triples to {args.out}")


if __name__ == "__main__":
    main()
