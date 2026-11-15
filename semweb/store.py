"""Read-only SPARQL over an in-memory Oxigraph store.

Oxigraph has no query timeout, so queries run in worker processes that hold their own copy of
the store; a query that runs past the time limit (cross joins, `*` paths, ORDER BY over a huge
result) is killed together with its worker, which is respawned on the next query.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import re
import threading
from multiprocessing.connection import Connection
from pathlib import Path

import pyoxigraph as ox

MAX_ROWS = 500
QUERY_TIMEOUT_S = float(os.environ.get("SPARQL_TIMEOUT_S", "5"))
WORKERS = int(os.environ.get("SPARQL_WORKERS", "2"))
WORKER_MEMORY_MB = int(os.environ.get("SPARQL_WORKER_MEMORY_MB", "2048"))
_START_TIMEOUT_S = 60.0
_COMMENT = re.compile(r"(?m)^\s*#.*$")
_READ = re.compile(r"^\s*((PREFIX\s+\S*:\s*<[^>]*>|BASE\s+<[^>]*>)\s*)*(SELECT|ASK)\b", re.I)
_SERVICE = re.compile(r"\bSERVICE\b", re.I)


class SparqlError(ValueError):
    pass


class SparqlBusy(SparqlError):
    """Every worker is busy; the caller should retry later."""


def _term(t: ox.NamedNode | ox.BlankNode | ox.Literal) -> dict:
    if isinstance(t, ox.Literal):
        return {"type": "literal", "value": t.value, "datatype": t.datatype.value}
    return {"type": "uri" if isinstance(t, ox.NamedNode) else "bnode", "value": t.value}


def _load(data: bytes) -> ox.Store:
    store = ox.Store()
    store.load(data, format=ox.RdfFormat.TURTLE)
    return store


def _evaluate(store: ox.Store, text: str) -> dict:
    try:
        result = store.query(text)
    except SyntaxError as e:
        raise SparqlError(f"invalid SPARQL: {e}") from e
    if isinstance(result, ox.QueryBoolean):
        return {"boolean": bool(result)}
    names = [v.value for v in result.variables]
    rows, truncated = [], False
    for i, solution in enumerate(result):
        if i == MAX_ROWS:
            truncated = True
            break
        rows.append({n: _term(solution[n]) for n in names if solution[n] is not None})
    return {"vars": names, "rows": rows, "truncated": truncated}


def _serve(conn: Connection, data: bytes, memory_mb: int) -> None:
    if memory_mb > 0:
        try:
            import resource

            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError):
            pass
    store = _load(data)
    conn.send(None)
    while True:
        try:
            text = conn.recv()
        except EOFError:
            return
        try:
            conn.send(("ok", _evaluate(store, text)))
        except SparqlError as e:
            conn.send(("error", str(e)))
        except Exception as e:  # e.g. MemoryError, OSError from the store
            conn.send(("error", f"query failed: {type(e).__name__}"))


class _Worker:
    def __init__(self, data: bytes, memory_mb: int) -> None:
        self._data = data
        self._memory_mb = memory_mb
        self._proc: mp.process.BaseProcess | None = None
        self._conn: Connection | None = None

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_serve, args=(child, self._data, self._memory_mb), daemon=True
        )
        self._proc.start()
        child.close()
        try:
            if not self._conn.poll(_START_TIMEOUT_S):
                raise EOFError
            self._conn.recv()
        except (EOFError, OSError) as e:
            self.stop()
            raise SparqlError("SPARQL worker failed to start") from e

    def run(self, text: str, timeout: float) -> dict:
        if self._proc is None or not self._proc.is_alive():
            self.stop()
            self._start()
        assert self._conn is not None
        try:
            self._conn.send(text)
            if not self._conn.poll(timeout):
                self.stop()
                raise SparqlError(f"query exceeded the {timeout:g} s time limit")
            status, payload = self._conn.recv()
        except (EOFError, OSError) as e:  # worker died, e.g. hit its memory limit
            self.stop()
            raise SparqlError("query failed: too expensive") from e
        if status == "error":
            raise SparqlError(payload)
        return payload

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join()
            self._proc = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class RdfStore:
    def __init__(
        self,
        turtle: str | bytes,
        *,
        timeout: float = QUERY_TIMEOUT_S,
        workers: int = WORKERS,
        memory_mb: int = WORKER_MEMORY_MB,
    ) -> None:
        data = turtle.encode() if isinstance(turtle, str) else turtle
        self._len = len(_load(data))
        self._timeout = timeout
        self._all = [_Worker(data, memory_mb) for _ in range(max(1, workers))]
        self._idle: queue.Queue[_Worker] = queue.Queue()
        for w in self._all:
            self._idle.put(w)
        self._closed = threading.Event()

    @classmethod
    def from_path(cls, path: Path, **kwargs) -> RdfStore:
        return cls(path.read_bytes(), **kwargs)

    def __len__(self) -> int:
        return self._len

    def query(self, query: str) -> dict:
        """SELECT → {vars, rows, truncated}; ASK → {boolean}. Anything else is rejected."""
        text = _COMMENT.sub("", query)
        if not _READ.match(text):
            raise SparqlError("only SELECT and ASK queries are allowed")
        if _SERVICE.search(text):
            raise SparqlError("federated SERVICE queries are not allowed")
        if self._closed.is_set():
            raise SparqlBusy("SPARQL store is closed")
        try:
            worker = self._idle.get(timeout=self._timeout)
        except queue.Empty:
            raise SparqlBusy("SPARQL workers are busy; try again later") from None
        try:
            return worker.run(text, self._timeout)
        finally:
            self._idle.put(worker)

    def close(self) -> None:
        self._closed.set()
        for w in self._all:
            w.stop()
