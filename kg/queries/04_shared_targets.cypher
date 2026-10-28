// name: shared_targets
// description: Protein targets two drugs have in common.
// params: {"drug_a": "Epinephrine", "drug_b": "Norepinephrine"}
MATCH (a:Drug {name: $drug_a})-[:TARGET]->(g:Gene)<-[:TARGET]-(b:Drug {name: $drug_b})
RETURN g.name AS gene
ORDER BY gene
