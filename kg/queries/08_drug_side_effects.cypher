// name: drug_side_effects
// description: Side effects (phenotypes) recorded for a drug.
// params: {"drug": "Prednisone"}
MATCH (d:Drug {name: $drug})-[:DRUG_EFFECT]->(p:Phenotype)
RETURN p.name AS effect
ORDER BY effect
LIMIT 50
