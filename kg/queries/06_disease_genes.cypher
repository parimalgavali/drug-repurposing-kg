// name: disease_genes
// description: Genes/proteins associated with a disease.
// params: {"disease": "Parkinson disease"}
MATCH (x:Disease {name: $disease})-[:DISEASE_PROTEIN]->(g:Gene)
RETURN g.name AS gene, g.source_id AS ncbi_gene_id
ORDER BY gene
