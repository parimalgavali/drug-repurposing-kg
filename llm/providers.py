"""LLM providers behind one small protocol. Tests and CI use `RecordedProvider` only.

`LLM_PROVIDER=anthropic|ollama|recorded` selects the provider; `LLM_RECORD=1` wraps a live
provider so every response is also written to the fixture directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
FIXTURES = Path("tests/fixtures/llm")


class Provider(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]: ...

    def structured(self, system: str, prompt: str, schema: type[T]) -> T: ...


def strict_schema(schema: type[BaseModel]) -> dict:
    """JSON schema with additionalProperties: false on every object, as structured outputs need."""

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for v in node.values():
                close(v)
        elif isinstance(node, list):
            for v in node:
                close(v)
        return node

    return close(schema.model_json_schema())


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, client=None) -> None:
        import anthropic

        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            yield from s.text_stream

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        # Native structured outputs: the response text is JSON constrained to the schema.
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": strict_schema(schema)}},
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return schema.model_validate_json(text)


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def _chat(self, system: str, prompt: str, **extra) -> dict:
        r = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **extra,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return self._chat(system, prompt)["message"]["content"]

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        yield self.complete(system, prompt, max_tokens)

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        out = self._chat(system, prompt, format=schema.model_json_schema())
        return schema.model_validate_json(out["message"]["content"])


class MissingRecording(KeyError):
    pass


class RecordedProvider:
    """Replays responses keyed by a hash of (kind, system, prompt, schema); records via `inner`."""

    name = "recorded"

    def __init__(self, fixtures: Path = FIXTURES, inner: Provider | None = None) -> None:
        self.fixtures = fixtures
        self.inner = inner

    @staticmethod
    def key(kind: str, system: str, prompt: str, schema: str = "") -> str:
        blob = json.dumps([kind, system, prompt, schema], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _replay(self, kind: str, system: str, prompt: str, schema: str, live) -> object:
        path = self.fixtures / f"{self.key(kind, system, prompt, schema)}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        if self.inner is None:
            raise MissingRecording(f"no recording {path.name} for: {prompt[:80]!r}")
        response = live()
        self.fixtures.mkdir(parents=True, exist_ok=True)
        record = {"kind": kind, "schema": schema, "prompt": prompt[:400], "response": response}
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
        return response

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return self._replay(
            "complete", system, prompt, "", lambda: self.inner.complete(system, prompt, max_tokens)
        )

    def stream(self, system: str, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        text = self.complete(system, prompt, max_tokens)
        for i in range(0, len(text), 40):
            yield text[i : i + 40]

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        data = self._replay(
            "structured",
            system,
            prompt,
            schema.__name__,
            lambda: self.inner.structured(system, prompt, schema).model_dump(mode="json"),
        )
        return schema.model_validate(data)


def get_provider(name: str | None = None) -> Provider:
    name = name or os.environ.get("LLM_PROVIDER", "recorded")
    live: dict[str, type] = {"anthropic": AnthropicProvider, "ollama": OllamaProvider}
    if name == "recorded":
        return RecordedProvider()
    provider = live[name]()
    if os.environ.get("LLM_RECORD") == "1":
        return RecordedProvider(inner=provider)
    return provider
