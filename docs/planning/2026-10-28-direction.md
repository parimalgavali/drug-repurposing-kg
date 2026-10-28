# RepurposeKG: Planning & Investigation (v0.1)

Date: 2026-10-28. Status: planning only. Nothing is implemented yet.

## 1. Goal

A portfolio project for AI / data / backend roles in Germany (pharma, biotech, health-tech, and any team that uses knowledge graphs). It covers three skills that are in demand right now, and every one of them is backed by real code and real data:

1. **Knowledge graph engineering**: load a real biomedical KG with 129k nodes and 4M relationships into Neo4j and query it with Cypher.
2. **Graph machine learning**: predict missing *drug → disease* "indication" links, measured honestly against baselines on a disease-held-out split.
3. **GraphRAG**: an LLM answers natural-language questions by linking entities, running *validated, read-only* Cypher, and citing the graph paths it used.

It also fits the user's background: an M.Sc. in Informatics for Natural Sciences, plus the earlier clinical decision-support project (`cdss_project`).

## 2. The one user loop

A researcher opens one screen and either:

- **picks a disease** (e.g. "Parkinson disease"). They get a ranked list of candidate drugs that are *not* already indicated for it. Each candidate shows a model score, 2–3 explanation paths through the graph (drug → target gene → pathway → disease gene), and a badge with the number of ClinicalTrials.gov studies that already test that pair; or
- **asks a question** ("Which drugs target genes associated with both asthma and psoriasis?"). They get a short answer, the Cypher that produced it, and the subgraph as a picture.

There are no dashboards, logins or admin pages.

## 3. Data: verified 2026-10-28

| Fact | Value |
| --- | --- |
| Dataset | **PrimeKG** (Chandak, Huang & Zitnik, *Scientific Data* 2023), Harvard Dataverse `doi:10.7910/DVN/IXA7BM`, version 2 (2022-05-02) |
| License label | Dataverse: **CC0 1.0**. Code repo (`mims-harvard/PrimeKG`): MIT. Caveat: it integrates 20 sources. DrugBank-derived drug content is CC BY-NC 4.0, so this project is non-commercial and says so |
| Download | `kg.csv` 982 MB (sha256 `0a5a010f…1cfb5c69`), `nodes.tab` 8.9 MB, `drug_features.tab` 10 MB, `disease_features.tab` 114 MB. Plain HTTPS, no login, ~7 MB/s from the dev VM |
| Size | 129,375 nodes in 10 types; `kg.csv` has 8,100,498 rows = 4,050,249 undirected edges stored in both directions, in 30 relation types |
| Nodes | gene/protein 27,671 · biological_process 28,642 · disease 17,080 · effect/phenotype 15,311 · anatomy 14,035 · molecular_function 11,169 · drug 7,957 · cellular_component 4,176 · pathway 2,516 · exposure 818 |
| Target relations | `indication` 9,388 · `contraindication` 30,675 · `off-label use` 2,568 (undirected counts) |
| Largest relations | drug–drug synergy 2.67M rows, anatomy–protein expression 3.04M rows. Both are candidates to drop from the *training* graph to keep CPU training feasible |
| Text | `drug_features` (DrugBank description, mechanism, ATC codes, …), `disease_features` (MONDO/UMLS/Orphanet definitions, Mayo Clinic symptom text) |

Second dataset for external validation: the **ClinicalTrials.gov API v2** (`/api/v2/studies?query.cond=…&query.intr=…`). It's public with no key, and I checked that it responds. Hetionet (CC0 with per-edge licenses, 2017) is a smaller alternative, kept as a stretch cross-check.

## 4. Prior work and how we position against it

TxGNN (Huang et al., *Nature Medicine* 2024, MIT code) trains a GNN on PrimeKG for zero-shot drug repurposing. **We do not claim to beat it.** We rebuild a smaller, readable version end-to-end: KG in a real graph database, baselines vs GNN on a disease-held-out split, explanations as graph paths, an LLM interface, and a usable UI. Where possible we report TxGNN's published numbers next to ours, with the differences in setup stated.

## 5. Stack (details in `DECISIONS.md`)

Python 3.12 + uv · Neo4j 5 Community (Cypher, full-text index) · PyTorch Geometric (heterogeneous GNN, CPU) + PyKEEN (KG-embedding baselines) · FastAPI + SSE · Anthropic Claude via a provider interface with an Ollama fallback · Vue 3 + TypeScript + Cytoscape.js · GitHub Actions.

## 6. Risks

| Risk | Mitigation |
| --- | --- |
| GNN training on 4M edges on CPU is too slow | Train on a pruned graph (drop drug–drug synergy and anatomy expression, ~1.9M edges), use mini-batch `LinkNeighborLoader` and small hidden sizes, and record timings in the model card |
| The demo makes medical claims | Prominent "research demo, not medical advice" notice; predictions are labelled hypotheses; trial counts are shown, not interpreted |
| LLM-generated Cypher is wrong or unsafe | Templates for the top intents; generated Cypher runs only in a read-only transaction after `EXPLAIN`, with an allowlist of labels/relations, a forced `LIMIT` and a timeout; measured on a gold question set |
| Name matching to ClinicalTrials.gov is noisy | Use exact drug names and disease names from PrimeKG, report it as a weak signal, cache responses, and compare against a random-pair baseline |
| 1 GB dataset in CI | CI uses a committed ~2k-edge fixture subgraph; the full data is only fetched locally |

## 7. Out of scope

Auth, multi-user, cloud deployment, training on GPUs, building our own KG from literature (NER/relation extraction), clinical use.
