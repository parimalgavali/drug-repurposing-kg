"""Thin Neo4j driver wrapper. `KGClient.read` only ever runs read-only transactions."""

from __future__ import annotations

import os
from typing import Any

from neo4j import READ_ACCESS, Driver, GraphDatabase, ManagedTransaction, unit_of_work


def connect(uri: str | None = None, user: str | None = None, password: str | None = None) -> Driver:
    return GraphDatabase.driver(
        uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            user or os.environ.get("NEO4J_USER", "neo4j"),
            password or os.environ.get("NEO4J_PASSWORD", ""),
        ),
    )


class KGClient:
    def __init__(self, driver: Driver | None = None, database: str | None = None) -> None:
        self._driver = driver or connect()
        self.database = database or os.environ.get("NEO4J_DATABASE", "neo4j")

    def read(
        self, query: str, params: dict[str, Any] | None = None, timeout: float = 10.0
    ) -> list[dict[str, Any]]:
        @unit_of_work(timeout=timeout)
        def work(tx: ManagedTransaction) -> list[dict[str, Any]]:
            return tx.run(query, params or {}).data()

        with self._driver.session(database=self.database, default_access_mode=READ_ACCESS) as s:
            return s.execute_read(work)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> KGClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
