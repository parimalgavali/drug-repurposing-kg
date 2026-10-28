// name: drugs_indicated_for_disease
// description: Approved indications: drugs with an INDICATION edge to the disease.
// params: {"disease": "asthma"}
MATCH (d:Drug)-[:INDICATION]->(x:Disease {name: $disease})
RETURN d.name AS drug, d.source_id AS drugbank_id
ORDER BY drug
