"""Apply kg/schema.cypher and wait for indexes to come online: `uv run python -m kg.schema`."""

from __future__ import annotations

from pathlib import Path

from neo4j import Driver

from kg.client import connect

SCHEMA = Path(__file__).with_name("schema.cypher")


def statements(text: str) -> list[str]:
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return [s.strip() for s in body.split(";") if s.strip()]


def apply_schema(driver: Driver, database: str = "neo4j", path: Path = SCHEMA) -> int:
    stmts = statements(path.read_text())
    with driver.session(database=database) as session:
        for stmt in stmts:
            session.run(stmt).consume()
        session.run("CALL db.awaitIndexes(900)").consume()
    return len(stmts)


if __name__ == "__main__":
    with connect() as driver:
        print(f"applied {apply_schema(driver)} schema statements")
