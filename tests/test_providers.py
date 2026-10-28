import pytest
from pydantic import BaseModel

from graphrag.eval import matches
from llm import RecordedProvider
from llm.providers import MissingRecording, strict_schema


class Pick(BaseModel):
    drug: str
    disease: str | None = None


class Fake:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, system, prompt, max_tokens=1024):
        self.calls += 1
        return f"echo {prompt}"

    def stream(self, system, prompt, max_tokens=1024):
        yield self.complete(system, prompt, max_tokens)

    def structured(self, system, prompt, schema):
        self.calls += 1
        return schema(drug="Salbutamol", disease="asthma")


def test_record_then_replay(tmp_path):
    inner = Fake()
    rec = RecordedProvider(tmp_path, inner=inner)
    assert rec.complete("s", "hi") == "echo hi"
    assert rec.structured("s", "why", Pick) == Pick(drug="Salbutamol", disease="asthma")
    assert inner.calls == 2 and len(list(tmp_path.glob("*.json"))) == 2

    replay = RecordedProvider(tmp_path)
    assert replay.complete("s", "hi") == "echo hi"
    assert "".join(replay.stream("s", "hi")) == "echo hi"
    assert replay.structured("s", "why", Pick).drug == "Salbutamol"


def test_missing_recording_raises(tmp_path):
    with pytest.raises(MissingRecording):
        RecordedProvider(tmp_path).complete("s", "never recorded")


def test_key_depends_on_every_part():
    k = RecordedProvider.key
    assert (
        len({k("complete", "s", "p"), k("complete", "t", "p"), k("structured", "s", "p", "X")}) == 3
    )


def test_strict_schema_closes_objects():
    schema = strict_schema(Pick)
    assert schema["additionalProperties"] is False


def test_matches_ignores_column_names_and_extra_columns():
    gold = [{"id": 1}, {"id": 2}]
    assert matches(gold, ["id"], [{"drug_id": 2, "name": "b"}, {"drug_id": 1, "name": "a"}])
    assert not matches(gold, ["id"], [{"drug_id": 1}])
    assert matches([{"n": 40}], ["n"], [{"drug_count": 40.0}])
    assert matches([], ["id"], [])
