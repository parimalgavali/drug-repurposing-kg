/** Static-snapshot backend for the GitHub Pages build (`VITE_STATIC=1`): reads the files written
 * by `scripts/export_static.py`, searches in the browser and asks ClinicalTrials.gov directly. */
import { type Lang, t } from "../i18n";
import type { Candidates, Explanation, SearchHit, Trials } from "./client";
import type { SseEvent } from "./sse";

const DATA = `${import.meta.env.BASE_URL}data`;
const CTGOV = "https://clinicaltrials.gov/api/v2/studies";
const PHASES: Record<string, number> = { EARLY_PHASE1: 0.5, PHASE1: 1, PHASE2: 2, PHASE3: 3, PHASE4: 4 };

async function json<T>(path: string, missing: string): Promise<T> {
  const r = await fetch(`${DATA}/${path}`);
  if (!r.ok) throw new Error(missing);
  return (await r.json()) as T;
}

type Entry = [id: number, name: string, nameDe: string | null];
let index: Promise<Entry[]> | null = null;

function match(name: string | null, needle: string): number {
  if (!name) return 0;
  const n = name.toLowerCase();
  if (n === needle) return 1;
  if (n.startsWith(needle)) return 0.9;
  if (n.includes(` ${needle}`)) return 0.85;
  return n.includes(needle) ? 0.8 : 0;
}

export async function search(q: string, limit: number): Promise<SearchHit[]> {
  index ??= json<Entry[]>("search.json", t("snapshotMissing"));
  const needle = q.toLowerCase();
  const hits: SearchHit[] = [];
  for (const [id, name, de] of await index) {
    const confidence = Math.max(match(name, needle), match(de, needle));
    if (confidence) hits.push({ id, name, name_de: de, label: "Disease", confidence });
  }
  hits.sort((a, b) => b.confidence - a.confidence || a.name.length - b.name.length);
  return hits.slice(0, limit);
}

export const candidates = (disease: number) =>
  json<Candidates>(`candidates/${disease}.json`, t("snapshotMissing"));

export const explain = (drug: number, disease: number) =>
  json<Explanation>(`explain/${drug}_${disease}.json`, t("notInSnapshot"));

interface Study {
  protocolSection?: { statusModule?: { overallStatus?: string }; designModule?: { phases?: string[] } };
}

export async function trials(drug: string, disease: string): Promise<Trials> {
  const params = new URLSearchParams({
    "query.cond": disease,
    "query.intr": drug,
    countTotal: "true",
    pageSize: "100",
    fields: "NCTId,OverallStatus,Phase",
  });
  const r = await fetch(`${CTGOV}?${params}`);
  if (!r.ok) throw new Error(`${r.status}`);
  const body = (await r.json()) as { totalCount?: number; studies?: Study[] };
  const statuses: Record<string, number> = {};
  const phases: number[] = [];
  for (const s of body.studies ?? []) {
    const status = s.protocolSection?.statusModule?.overallStatus;
    if (status) statuses[status] = (statuses[status] ?? 0) + 1;
    for (const p of s.protocolSection?.designModule?.phases ?? []) if (p in PHASES) phases.push(PHASES[p]);
  }
  return {
    studies: body.totalCount ?? 0,
    max_phase: phases.length ? Math.max(...phases) : null,
    statuses,
    url: `https://clinicaltrials.gov/search?${new URLSearchParams({ cond: disease, intr: drug })}`,
    cached: false,
  };
}

export async function ask(question: string, lang: Lang, onEvent: (e: SseEvent) => void): Promise<void> {
  const answers = await json<Record<string, Record<string, string>>>("answers/index.json", t("askStatic"));
  const file = answers[lang]?.[question.trim()];
  if (!file) throw new Error(t("askStatic"));
  for (const e of await json<SseEvent[]>(`answers/${file}`, t("askStatic"))) {
    if (e.event === "token") await new Promise((done) => setTimeout(done, 15));
    onEvent(e);
  }
}
