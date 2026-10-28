import json
from pathlib import Path

import httpx

from validation.ctgov import CTGov, summarise

FIXTURES = Path(__file__).parent / "fixtures" / "ctgov"
RECORDED = {
    "cabergoline": json.loads((FIXTURES / "cabergoline_parkinson.json").read_text()),
    "sotagliflozin": json.loads((FIXTURES / "sotagliflozin_parkinson.json").read_text()),
}


def _client(calls: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return httpx.Response(200, json=RECORDED[request.url.params["query.intr"].lower()])

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_summarise_recorded_response():
    s = summarise("Parkinson disease", "Cabergoline", RECORDED["cabergoline"])
    assert s.studies == 7 and s.max_phase == 4.0
    assert s.statuses == {"COMPLETED": 6, "TERMINATED": 1}
    empty = summarise("Parkinson disease", "Sotagliflozin", RECORDED["sotagliflozin"])
    assert empty.studies == 0 and empty.max_phase is None and empty.statuses == {}


def test_query_parameters_and_cache_hit(tmp_path):
    calls: list = []
    cache = tmp_path / "ct.jsonl"
    ct = CTGov(cache=cache, min_interval=0, client=_client(calls))
    first = ct.lookup("Parkinson disease", "Cabergoline")
    again = ct.lookup("parkinson disease ", "CABERGOLINE")
    assert first == again and len(calls) == 1
    assert calls[0]["query.cond"] == "Parkinson disease" and calls[0]["countTotal"] == "true"

    reloaded = CTGov(cache=cache, min_interval=0, client=_client(calls))
    assert reloaded.lookup("Parkinson disease", "Cabergoline") == first
    assert len(calls) == 1 and reloaded.requests == 0


def test_rejected_query_counts_as_no_study(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400)))
    ct = CTGov(cache=tmp_path / "ct.jsonl", min_interval=0, client=client)
    s = ct.lookup("galactosemia", "N-[(2R,3S)-3-amino]-benzenesulfonamide")
    assert s.studies == 0 and s.max_phase is None and ct.rejected == 1
