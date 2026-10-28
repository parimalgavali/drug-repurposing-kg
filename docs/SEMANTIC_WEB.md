# Semantic-web view: RDF, SPARQL, SHACL

The served graph lives in Neo4j (property graph, Cypher). This view exports a
**disease-centred subgraph** to RDF so the same data can be queried with SPARQL,
validated with SHACL and linked to the source ontologies.

```bash
make rdf DISEASE=33675        # asthma → data/rdf/export.ttl (needs a loaded Neo4j)
make api                      # serves /api/sparql over that file (Oxigraph, in memory)
curl -s localhost:8000/api/diseases/33675/rdf     # Turtle on the fly, no file needed
```

## What gets exported

For one disease: the disease, its top-k predicted drugs (`k=20`) and known drugs,
those drugs' gene targets, the disease's genes and its phenotypes (asthma: 1,286 triples).

| PrimeKG source | IRI | Class |
| --- | --- | --- |
| MONDO | `http://purl.obolibrary.org/obo/MONDO_0004979` | `rkg:Disease` |
| MONDO_grouped | `https://w3id.org/repurposekg/node/{id}` + `rkg:groups` → each MONDO IRI | `rkg:Disease` |
| DrugBank | `https://go.drugbank.com/drugs/DB00938` | `rkg:Drug` |
| NCBI Gene | `https://identifiers.org/ncbigene:154` | `rkg:Gene` |
| HPO | `http://purl.obolibrary.org/obo/HP_0030828` | `rkg:Phenotype` |

Relationships become predicates in `rkg:` (`https://w3id.org/repurposekg/vocab#`):
`INDICATION` → `rkg:indication`, `OFF_LABEL_USE` → `rkg:offLabelUse`, `TARGET` → `rkg:target`,
`DISEASE_PROTEIN` → `rkg:diseaseProtein`, `DISEASE_PHENOTYPE_POSITIVE` → `rkg:phenotype`.
Every entity keeps `rdfs:label`, `rkg:source` and `rkg:servedId` (the Neo4j/API id).

A model prediction has properties (score, rank, model version, trial count), which a
plain triple can't carry. It becomes an n-ary node rather than RDF-star, so any SPARQL 1.1
store can read it:

```turtle
pred:14263-33675 a rkg:Prediction ;
    rkg:drug drugbank:DB00938 ;
    rkg:disease obo:MONDO_0004979 ;
    rkg:score "4.997356"^^xsd:double ;
    rkg:rank 3 ;
    rkg:modelVersion "gnn-notext" .
```

## SHACL

[`semweb/shapes.ttl`](../semweb/shapes.ttl) checks that:
- every entity has exactly one label, served id and source;
- IRIs follow each ontology's pattern;
- grouped diseases list at least two MONDO members;
- predicates point at the right classes (a drug targets a `rkg:Gene`);
- each `rkg:Prediction` has exactly one drug, one disease, a `xsd:double` score and a rank ≥ 1.

`pyshacl` validates the asthma export as conforming. The tests also check that the
shapes reject a broken graph.

## `/api/sparql`

- **Method:** `GET /api/sparql?query=…` or `POST {"query": "…"}`.
- **Read-only:**
  - only `SELECT` and `ASK` are accepted;
  - `CONSTRUCT`, updates and federated `SERVICE` calls return 400;
  - Oxigraph's query API can't run updates in any case.
- **Limits:**
  - at most 500 rows come back (`truncated: true` beyond that);
  - the store holds only the exported subgraph, so queries stay small.
- **No file:** if no export file exists, the endpoint returns 503.

## Cypher vs SPARQL

Run on the full graph (asthma, `k=20`). Each pair returned identical results.

**Top predicted drugs** → Zileuton, Fluticasone propionate, Salmeterol, Flumethasone, Formoterol

```cypher
MATCH (d:Drug)-[p:PREDICTED_INDICATION]->(:Disease {name: 'asthma'})
RETURN d.name AS drug, p.score AS score ORDER BY p.rank LIMIT 5
```
```sparql
SELECT ?drug ?score WHERE {
  ?p a rkg:Prediction ; rkg:disease/rdfs:label "asthma" ;
     rkg:drug/rdfs:label ?drug ; rkg:score ?score ; rkg:rank ?rank .
} ORDER BY ?rank LIMIT 5
```

**Predicted drugs that target one of the disease's genes**
→ Arformoterol, Ephedrine, Formoterol, Montelukast, Salmeterol (1 shared gene each)

```cypher
MATCH (d:Drug)-[p:PREDICTED_INDICATION]->(s:Disease {name: 'asthma'}) WHERE p.rank <= 20
MATCH (d)-[:TARGET]-(g:Gene)-[:DISEASE_PROTEIN]-(s)
RETURN d.name AS drug, count(DISTINCT g) AS shared ORDER BY shared DESC, drug LIMIT 5
```
```sparql
SELECT ?drug (COUNT(DISTINCT ?gene) AS ?shared) WHERE {
  ?p a rkg:Prediction ; rkg:disease ?s ; rkg:drug ?d .
  ?d rdfs:label ?drug ; rkg:target ?gene .
  ?s rkg:diseaseProtein ?gene .
} GROUP BY ?drug ORDER BY DESC(?shared) ?drug LIMIT 5
```

**Is Salbutamol a known asthma drug?** → true

```cypher
RETURN EXISTS { MATCH (:Drug {name: 'Salbutamol'})-[:INDICATION]-(:Disease {name: 'asthma'}) }
```
```sparql
ASK { ?d rkg:indication ?s . ?s rdfs:label "asthma" . ?d rdfs:label "Salbutamol" }
```

The two languages differ in a few ways:

| | Cypher | SPARQL |
| --- | --- | --- |
| **Edge direction** | PrimeKG edges are undirected, and Cypher can match them either way. | Each triple has one direction. The export fixes one per predicate (drug → disease, drug → gene, disease → gene/phenotype). |
| **Edge properties** | Stored on the relationship itself. | Need an n-ary node such as `rkg:Prediction`. |
| **Identity** | Node ids are local to Neo4j. | IRIs are global, so the export links straight to MONDO, HPO, DrugBank and NCBI Gene, and other RDF data can be joined on them. |
| **Schema** | Mostly implicit: Neo4j only checks constraints and indexes. | SHACL states the expected shape of the data and validates it. |

## Licensing

Exports are derived from PrimeKG, which includes DrugBank content: non-commercial use
only (see [DATA.md](DATA.md)). `data/rdf/` is git-ignored, so no export is committed.
