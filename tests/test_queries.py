import re

from kg.queries import load_queries, parse_query
from kg.schema import SCHEMA, statements

WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|LOAD\s+CSV)\b", re.I)


def test_ten_documented_read_only_queries():
    queries = load_queries()
    assert len(queries) == 10
    assert len({q.name for q in queries}) == 10
    for q in queries:
        assert q.description and q.params, q.name
        assert not WRITE.search(q.cypher), q.name
        for key in q.params:
            assert f"${key}" in q.cypher, (q.name, key)


def test_parse_query_header():
    q = parse_query('// name: x\n// description: y\n// params: {"a": 1}\nRETURN $a // trailing')
    assert (q.name, q.description, q.params) == ("x", "y", {"a": 1})
    assert q.cypher == "RETURN $a // trailing"


def test_schema_statements():
    stmts = statements(SCHEMA.read_text())
    assert sum(s.startswith("CREATE CONSTRAINT") for s in stmts) == 10
    assert any("FULLTEXT INDEX entityNames" in s for s in stmts)
