import pytest

from graphrag.text2cypher import MAX_LIMIT, CypherRejected, guard

REJECT = [
    "MATCH (n) DETACH DELETE n",
    "CREATE (:Drug {id: 1, name: 'x'})",
    "MATCH (d:Drug {id: 1}) SET d.name = 'pwned' RETURN d",
    "MERGE (a:Drug {id: 1})-[:INDICATION]->(b:Disease {id: 2})",
    "MATCH (d:Drug) REMOVE d.name RETURN d",
    "LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row",
    "CALL dbms.security.listUsers()",
    "CALL apoc.cypher.runWrite('MATCH (n) DELETE n', {})",
    "MATCH (d:Drug) RETURN d.name; MATCH (n) DELETE n",
    "MATCH (d:Medication)-[:TREATS]->(s:Disease) RETURN d.name",
    "MATCH (d:Drug)-[:TREATS]->(s:Disease) RETURN d.name",
    "MATCH (d:Drug) CALL { MATCH (n) DELETE n } RETURN d",
    "",
]


@pytest.mark.parametrize("query", REJECT)
def test_rejects(query):
    with pytest.raises(CypherRejected):
        guard(query)


def test_allows_reads_and_adds_limit():
    q = guard("MATCH (d:Drug)-[:INDICATION|OFF_LABEL_USE]-(s:Disease {id: 5}) RETURN d.name, d.id")
    assert q.endswith(f"LIMIT {MAX_LIMIT}")
    assert guard("MATCH (d:Drug) RETURN d.name LIMIT 500").endswith(f"LIMIT {MAX_LIMIT}")
    assert guard("MATCH (d:Drug) RETURN d.name LIMIT 5").endswith("LIMIT 5")


def test_keywords_inside_strings_and_fulltext_are_fine():
    q = "MATCH (d:Drug) WHERE d.name = 'Set Delete Create' RETURN d.name"
    assert guard(q).startswith(q)
    ft = "CALL db.index.fulltext.queryNodes('entityNames', 'asthma') YIELD node RETURN node.name"
    assert guard(ft).startswith(ft)
    assert guard("```cypher\nMATCH (g:Gene) RETURN g.name\n```").startswith("MATCH (g:Gene)")
