// name: similar_diseases_by_phenotype
// description: Diseases sharing the most positive phenotypes with the given disease.
// params: {"disease": "Parkinson disease"}
MATCH (x:Disease {name: $disease})-[:DISEASE_PHENOTYPE_POSITIVE]->(p:Phenotype)<-[:DISEASE_PHENOTYPE_POSITIVE]-(y:Disease)
WHERE y <> x
RETURN y.name AS disease, count(p) AS shared_phenotypes, collect(p.name)[..5] AS examples
ORDER BY shared_phenotypes DESC, disease
LIMIT 10
