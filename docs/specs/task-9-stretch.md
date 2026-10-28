# Task 9: Stretch (only after task 8)

- **Semantic-web view:** export a disease-centred subgraph to RDF (Turtle) with stable IRIs (MONDO, DrugBank, NCBI Gene, HPO PURLs), load it into Oxigraph, add a `/api/sparql` read-only endpoint and SHACL shapes that validate it (`pyshacl`). Document Cypher vs SPARQL side by side.
- **German labels:** fetch German names for diseases and drugs from Wikidata (MONDO id P5270, DrugBank id P715) and add a DE/EN toggle in the UI and answers.
- **Hetionet cross-check:** evaluate the served model's top predictions against Hetionet's `TREATS` edges that are absent from PrimeKG.
- **GNNExplainer** comparison with metapath explanations on 20 pairs.
