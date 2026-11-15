import json
import threading
import time
from pathlib import Path

import httpx

from validation.ctgov import CTGov, search_url, summarise

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


def test_cached_and_search_url(tmp_path):
    ct = CTGov(cache=tmp_path / "ct.jsonl", min_interval=0, client=_client([]))
    assert ct.cached("Parkinson disease", "Cabergoline") is None
    ct.lookup("Parkinson disease", "Cabergoline")
    assert ct.cached("parkinson disease", "cabergoline").studies == 7
    assert search_url("Parkinson disease", "L-Dopa") == (
        "https://clinicaltrials.gov/search?cond=Parkinson+disease&intr=L-Dopa"
    )


def test_unwritable_cache_keeps_the_in_memory_result(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    calls: list = []
    ct = CTGov(cache=blocker / "ct.jsonl", min_interval=0, client=_client(calls))
    assert ct.lookup("Parkinson disease", "Cabergoline").studies == 7
    assert ct.lookup("Parkinson disease", "Cabergoline").studies == 7 and len(calls) == 1


def test_lookups_wait_for_the_rate_limit_outside_the_lock(tmp_path):
    gate, calls = threading.Event(), []
    inner = _client(calls)

    def handler(request: httpx.Request) -> httpx.Response:
        gate.wait(5)
        return inner.get(str(request.url))

    ct = CTGov(cache=tmp_path / "ct.jsonl", min_interval=0.2, client=_client([]))
    ct.lookup("Parkinson disease", "Sotagliflozin")
    ct.client = httpx.Client(transport=httpx.MockTransport(handler))
    slow = [
        threading.Thread(target=ct.lookup, args=("Parkinson disease", "Cabergoline"))
        for _ in range(3)
    ]
    for t in slow:
        t.start()
    time.sleep(0.1)
    started = time.monotonic()
    assert ct.lookup("parkinson disease", "sotagliflozin").studies == 0
    assert time.monotonic() - started < 0.05  # cached hits don't queue behind HTTP or sleeps
    gate.set()
    for t in slow:
        t.join(5)
    assert len(calls) == 3 and ct.requests == 4
    assert len((tmp_path / "ct.jsonl").read_text().splitlines()) == 2  # one line per pair


def test_backoff_delays_the_next_slot_for_everyone(tmp_path):
    ct = CTGov(cache=tmp_path / "ct.jsonl", min_interval=0, client=_client([]))
    ct._back_off(0.3)
    assert 0.2 < ct._reserve() <= 0.3
