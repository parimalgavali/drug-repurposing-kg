// name: drug_protein_interactions
// description: Proteins a drug acts on, split by role (target, enzyme, transporter, carrier).
// params: {"drug": "Ibuprofen"}
MATCH (d:Drug {name: $drug})-[r:TARGET|ENZYME|TRANSPORTER|CARRIER]->(g:Gene)
RETURN type(r) AS role, g.name AS gene
ORDER BY role, gene
