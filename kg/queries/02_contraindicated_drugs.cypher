// name: contraindicated_drugs
// description: Drugs contraindicated for the disease.
// params: {"disease": "asthma"}
MATCH (d:Drug)-[:CONTRAINDICATION]->(x:Disease {name: $disease})
RETURN d.name AS drug, d.source_id AS drugbank_id
ORDER BY drug
