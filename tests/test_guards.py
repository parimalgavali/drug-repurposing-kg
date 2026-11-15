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
    "MATCH p=(a)-[*1..10]-(b) RETURN count(p)",
    "MATCH (a:Drug)-[:TARGET*]->(b) RETURN b.name",
    "MATCH (a:Drug)-[r:TARGET|ENZYME *2..]-(b) RETURN b.name",
    "MATCH p = shortestPath((a:Drug)-[*]-(b:Disease)) RETURN p",
    "MATCH (a:Drug)-[:TARGET]-{1,5}(b) RETURN b.name",
    "MATCH (a:Drug)-->+(b) RETURN b.name",
    "MATCH ((a)-[:PROTEIN_PROTEIN]-(b)){1,10} RETURN count(*)",
    "MATCH ((a)-[:PROTEIN_PROTEIN]-(b))+ RETURN count(*)",
    "MATCH (a:Drug)-[r:TARGET {w: [1]} *1..5]-(b) RETURN b.name",
    "MATCH (d:Drug) OPTIONAL MATCH ((a)-[:PROTEIN_PROTEIN]-(b))+ RETURN count(*)",
    "MATCH ((a)--(b)){1,3} UNWIND [1] AS x RETURN x",
    "MATCH (a:Drug)-[`r)` *1..10]-(b) RETURN b.name",
    "MATCH (a)-[`x'`]-(b) MATCH (c)-[*1..9]-(d) RETURN `y'`",
    "MATCH (d:Drug) RETURN d.name AS n UNION MATCH (g:Gene) RETURN g.name AS n",
    "MATCH (d:Drug) RETURN d.name AS n union all MATCH (a)--(b)--(c) RETURN a.name AS n LIMIT 5",
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


def test_star_outside_paths_is_fine():
    for q in (
        "MATCH (d:Drug)-[:TARGET]-(g:Gene) RETURN d.name, count(*) AS n ORDER BY n DESC",
        "MATCH (d:Drug)-[:TARGET]-(g:Gene) RETURN d.name, count(g) * 2 AS n",
        "MATCH (d:Drug) RETURN [x IN range(1, 3) | x * d.id] AS xs",
        "MATCH (d:Drug) RETURN count(d) * (2) AS a, count(d) + (2) AS b, (d.id - 1) * 2 AS c",
        "MATCH (d:Drug) WHERE d.name = 'Union [*1..10]' RETURN *",
        "MATCH (d:`Drug`)-[:`TARGET`]-(g) WHERE d.url = 'http://x' RETURN d.name AS `a b`, d.id",
    ):
        assert guard(q).startswith(q)


def test_keywords_inside_strings_and_fulltext_are_fine():
    q = "MATCH (d:Drug) WHERE d.name = 'Set Delete Create' RETURN d.name"
    assert guard(q).startswith(q)
    ft = "CALL db.index.fulltext.queryNodes('entityNames', 'asthma') YIELD node RETURN node.name"
    assert guard(ft).startswith(ft)
    assert guard("```cypher\nMATCH (g:Gene) RETURN g.name\n```").startswith("MATCH (g:Gene)")
