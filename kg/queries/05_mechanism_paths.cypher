// name: mechanism_paths
// description: Drug → protein (→ interacting protein) ← disease paths of at most 3 hops.
// params: {"drug": "Ibuprofen", "disease": "rheumatoid arthritis"}
MATCH (d:Drug {name: $drug}), (x:Disease {name: $disease})
MATCH p = (d)-[:TARGET|ENZYME|TRANSPORTER|CARRIER]->(:Gene)-[:PROTEIN_PROTEIN*0..1]-(:Gene)<-[:DISEASE_PROTEIN]-(x)
RETURN [n IN nodes(p) | n.name] AS path, [r IN relationships(p) | type(r)] AS relations, length(p) AS hops
ORDER BY hops
LIMIT 25
