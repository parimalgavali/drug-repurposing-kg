// name: entity_search
// description: Full-text search over node names across all labels (Lucene syntax).
// params: {"q": "parkinson*"}
CALL db.index.fulltext.queryNodes('entityNames', $q) YIELD node, score
RETURN labels(node)[0] AS label, node.name AS name, node.id AS id, score
ORDER BY score DESC
LIMIT 10
