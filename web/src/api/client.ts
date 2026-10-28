import type { components } from "./schema";
import type { Lang } from "../i18n";
import { type SseEvent, SseParser } from "./sse";

type Schemas = components["schemas"];
export type SearchHit = Schemas["SearchHit"];
export type Candidates = Schemas["Candidates"];
export type Candidate = Schemas["Candidate"];
export type Explanation = Schemas["Explanation"];
export type Cytoscape = Schemas["Cytoscape"];
export type Trials = Schemas["Trials"];
/** A candidate's on-demand ClinicalTrials.gov lookup. */
export type TrialState = "loading" | "error" | Trials;

export interface Entity {
  slot: string;
  mention: string;
  id?: number;
  name?: string;
  label?: string;
  confidence?: number;
}

export interface AskEvents {
  entities: Entity[];
  cypher: { queries: string[]; attempts: { cypher: string | null; error: string | null }[] };
  rows: { rows: Record<string, unknown>[]; total: number };
  token: { text: string };
  subgraph: Cytoscape;
  done: { intent: string; clarification: boolean };
  error: { message: string; detail?: string };
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return (await r.json()) as T;
}

export const search = (q: string, type = "Disease", limit = 8) =>
  get<SearchHit[]>(`/api/search?${new URLSearchParams({ q, type, limit: String(limit) })}`);

export const candidates = (disease: number, limit = 20) =>
  get<Candidates>(`/api/diseases/${disease}/candidates?limit=${limit}`);

export const explain = (drug: number, disease: number) =>
  get<Explanation>(`/api/pairs/${drug}/${disease}/explain`);

export const trials = (drug: number, disease: number) =>
  get<Trials>(`/api/pairs/${drug}/${disease}/trials`);

export async function ask(
  question: string,
  lang: Lang,
  onEvent: (e: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch("/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, lang }),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`${r.status}: ${await r.text()}`);
  const parser = new SseParser();
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.push(value).forEach(onEvent);
  }
}
