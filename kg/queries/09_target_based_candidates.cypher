// name: target_based_candidates
// description: Naive repurposing candidates: drugs targeting the disease's proteins without an existing indication.
// params: {"disease": "rheumatoid arthritis"}
MATCH (x:Disease {name: $disease})-[:DISEASE_PROTEIN]->(g:Gene)<-[:TARGET]-(d:Drug)
WHERE NOT (d)-[:INDICATION]->(x)
RETURN d.name AS drug, count(DISTINCT g) AS shared_genes, collect(DISTINCT g.name)[..5] AS genes
ORDER BY shared_genes DESC, drug
LIMIT 20
