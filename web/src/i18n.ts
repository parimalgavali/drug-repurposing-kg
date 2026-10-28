import { ref, watch } from "vue";

export type Lang = "en" | "de";

const STRINGS = {
  en: {
    bannerStrong: "Research demo, not medical advice.",
    bannerRest: "Predictions are hypotheses from a graph model, not evidence that a drug is safe or effective.",
    tagline: "Drug-repurposing hypotheses from PrimeKG, explained with graph paths.",
    searchPlaceholder: "Search a disease, e.g. asthma",
    searchLabel: "Search a disease",
    candidates: "Candidate drugs",
    for: "for",
    loading: "Loading…",
    searchFirst: "Search a disease to see model-predicted candidates.",
    noPredictions: "No predictions stored for this disease.",
    known: "Known treatments in the graph",
    knownContext: "shown for context",
    offLabel: "off-label",
    indication: "indication",
    inTrials: "in {n} trial(s)",
    noTrials: "no trials found",
    evidenceGraph: "Evidence graph",
    answerSubgraph: "Answer subgraph",
    why: "Why",
    pickHint: "Pick a candidate or ask a question to see its subgraph.",
    pathsFound: "paths found",
    truncated: "(truncated)",
    paths: "paths",
    eg: "e.g.",
    noPath: "No 2–3 hop mechanistic path found.",
    askTitle: "Ask the graph",
    askPlaceholder: "Ask a question in English",
    ask: "Ask",
    thinking: "Thinking…",
    try: "Try:",
    linked: "Linked:",
    rows: "rows",
    showCypher: "Show Cypher",
    language: "Language",
  },
  de: {
    bannerStrong: "Forschungsdemo, keine medizinische Beratung.",
    bannerRest:
      "Vorhersagen sind Hypothesen eines Graph-Modells, kein Beleg dafür, dass ein Medikament sicher oder wirksam ist.",
    tagline: "Hypothesen zur Arzneimittel-Umwidmung aus PrimeKG, erklärt mit Graph-Pfaden.",
    searchPlaceholder: "Krankheit suchen (englischer Name), z. B. asthma",
    searchLabel: "Krankheit suchen",
    candidates: "Kandidaten-Medikamente",
    for: "für",
    loading: "Lädt…",
    searchFirst: "Suche eine Krankheit, um vom Modell vorhergesagte Kandidaten zu sehen.",
    noPredictions: "Für diese Krankheit sind keine Vorhersagen gespeichert.",
    known: "Bekannte Behandlungen im Graphen",
    knownContext: "zur Einordnung",
    offLabel: "Off-Label",
    indication: "Indikation",
    inTrials: "in {n} Studie(n)",
    noTrials: "keine Studien gefunden",
    evidenceGraph: "Evidenz-Graph",
    answerSubgraph: "Antwort-Teilgraph",
    why: "Warum",
    pickHint: "Wähle einen Kandidaten oder stelle eine Frage, um den Teilgraphen zu sehen.",
    pathsFound: "Pfade gefunden",
    truncated: "(gekürzt)",
    paths: "Pfade",
    eg: "z. B.",
    noPath: "Kein mechanistischer Pfad über 2–3 Schritte gefunden.",
    askTitle: "Frag den Graphen",
    askPlaceholder: "Frage auf Englisch oder Deutsch",
    ask: "Fragen",
    thinking: "Denkt nach…",
    try: "Beispiele:",
    linked: "Verknüpft:",
    rows: "Zeilen",
    showCypher: "Cypher anzeigen",
    language: "Sprache",
  },
} as const;

export type Key = keyof (typeof STRINGS)["en"];

const stored = typeof localStorage === "undefined" ? null : localStorage.getItem("lang");
export const lang = ref<Lang>(stored === "de" ? "de" : "en");

watch(
  lang,
  (value) => {
    if (typeof localStorage !== "undefined") localStorage.setItem("lang", value);
    if (typeof document !== "undefined") document.documentElement.lang = value;
  },
  { immediate: true },
);

export function t(key: Key, n?: number): string {
  const s: string = STRINGS[lang.value][key];
  return n === undefined ? s : s.replace("{n}", String(n)).replace(/\((\w+)\)/, n === 1 ? "" : "$1");
}

/** The German name when DE is selected and Wikidata has one, else the PrimeKG (English) name. */
export function nm(name: string, de?: string | null): string {
  return lang.value === "de" && de ? de : name;
}
