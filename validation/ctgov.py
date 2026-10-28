"""ClinicalTrials.gov API v2 lookups for (drug, disease) pairs, rate-limited and cached as JSONL.

A registered study is evidence that someone tested the idea, not that the drug works.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

API = "https://clinicaltrials.gov/api/v2/studies"
CACHE = Path("artifacts/ctgov_cache.jsonl")
PHASES = {"EARLY_PHASE1": 0.5, "PHASE1": 1.0, "PHASE2": 2.0, "PHASE3": 3.0, "PHASE4": 4.0}


@dataclass(frozen=True)
class TrialSummary:
    condition: str
    intervention: str
    studies: int
    max_phase: float | None
    statuses: dict[str, int]


def summarise(condition: str, intervention: str, payload: dict) -> TrialSummary:
    """`studies` is the API's total; phase and statuses come from the first page (≤ 100)."""
    phases: list[float] = []
    statuses: dict[str, int] = {}
    for study in payload.get("studies", []):
        section = study.get("protocolSection", {})
        status = section.get("statusModule", {}).get("overallStatus")
        if status:
            statuses[status] = statuses.get(status, 0) + 1
        phases += [
            PHASES[p] for p in section.get("designModule", {}).get("phases", []) if p in PHASES
        ]
    return TrialSummary(
        condition,
        intervention,
        int(payload.get("totalCount", 0)),
        max(phases, default=None),
        statuses,
    )


def _key(condition: str, intervention: str) -> str:
    return f"{condition.strip().lower()}\t{intervention.strip().lower()}"


class CTGov:
    def __init__(
        self,
        cache: Path = CACHE,
        min_interval: float = 1.3,
        client: httpx.Client | None = None,
    ) -> None:
        self.cache_path = cache
        self.min_interval = min_interval
        self.client = client or httpx.Client(timeout=30, headers={"User-Agent": "RepurposeKG"})
        self._last = 0.0
        self.requests = 0
        self.cache: dict[str, TrialSummary] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.cache[_key(row["condition"], row["intervention"])] = TrialSummary(**row)

    def lookup(self, condition: str, intervention: str) -> TrialSummary:
        key = _key(condition, intervention)
        if key in self.cache:
            return self.cache[key]
        params = {
            "query.cond": condition,
            "countTotal": "true",
            "pageSize": "100",
            "fields": "NCTId,OverallStatus,Phase",
        }
        if intervention:
            params["query.intr"] = intervention
        payload = self._get(params)
        summary = summarise(condition, intervention, payload)
        self.cache[key] = summary
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as f:
            f.write(json.dumps(asdict(summary)) + "\n")
        return summary

    def _get(self, params: dict, retries: int = 4) -> dict:
        for attempt in range(retries):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            response = self.client.get(API, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2**attempt * 5)
                continue
            response.raise_for_status()
            return response.json()
        response.raise_for_status()
        raise RuntimeError(f"CT.gov kept failing for {params}")
