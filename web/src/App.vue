<script setup lang="ts">
import { computed, ref } from "vue";
import {
  type Candidate,
  type Candidates,
  type Cytoscape,
  type Explanation,
  type SearchHit,
  candidates,
  explain,
} from "./api/client";
import AskPanel from "./components/AskPanel.vue";
import CandidateList from "./components/CandidateList.vue";
import DiseaseSearch from "./components/DiseaseSearch.vue";
import GraphView from "./components/GraphView.vue";
import { lang, nm, t } from "./i18n";

const list = ref<Candidates | null>(null);
const loading = ref(false);
const selected = ref<number | null>(null);
const evidence = ref<Explanation | null>(null);
const graph = ref<Cytoscape | null>(null);
const pair = ref<{ drug: string; drugDe?: string | null; disease: string; diseaseDe?: string | null } | null>(null);
const answerGraph = ref(false);
const focus = ref<string | null>(null);
const error = ref<string | null>(null);

async function pickDisease(hit: SearchHit) {
  loading.value = true;
  selected.value = null;
  evidence.value = null;
  error.value = null;
  try {
    list.value = await candidates(hit.id);
  } catch (e) {
    error.value = String(e);
  } finally {
    loading.value = false;
  }
}

async function pickCandidate(c: Candidate) {
  if (!list.value) return;
  selected.value = c.drug_id;
  error.value = null;
  try {
    evidence.value = await explain(c.drug_id, list.value.disease.id);
    graph.value = evidence.value.cytoscape;
    pair.value = {
      drug: c.drug,
      drugDe: c.drug_de,
      disease: list.value.disease.name,
      diseaseDe: list.value.disease.name_de,
    };
    answerGraph.value = false;
    focus.value = null;
  } catch (e) {
    error.value = String(e);
  }
}

const title = computed(() => {
  if (answerGraph.value) return t("answerSubgraph");
  if (!pair.value) return t("evidenceGraph");
  const p = pair.value;
  return `${t("why")} ${nm(p.drug, p.drugDe)} → ${nm(p.disease, p.diseaseDe)}?`;
});

function showAnswerGraph(g: Cytoscape) {
  graph.value = g;
  answerGraph.value = true;
  evidence.value = null;
}
</script>

<template>
  <header class="banner" role="note">
    <strong>{{ t("bannerStrong") }}</strong> {{ t("bannerRest") }}
  </header>
  <div class="lang" role="group" :aria-label="t('language')">
    <button type="button" :aria-pressed="lang === 'en'" @click="lang = 'en'">EN</button>
    <button type="button" :aria-pressed="lang === 'de'" @click="lang = 'de'">DE</button>
  </div>
  <main>
    <div class="intro">
      <h1>RepurposeKG</h1>
      <p class="muted">{{ t("tagline") }}</p>
      <DiseaseSearch @pick="pickDisease" />
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </div>
    <div class="grid">
      <CandidateList :data="list" :selected="selected" :loading="loading" @select="pickCandidate" />
      <div>
        <GraphView :graph="graph" :focus="focus" :title="title" @focus="focus = $event" />
        <section v-if="evidence" class="panel" data-testid="evidence">
          <h3>{{ evidence.paths_found }} {{ t("pathsFound") }}{{ evidence.truncated ? ` ${t("truncated")}` : "" }}</h3>
          <ol>
            <li v-for="m in evidence.metapaths" :key="m.metapath">
              <code>{{ m.metapath }}</code> <small>DWPC {{ m.dwpc.toPrecision(3) }} · {{ m.count }} {{ t("paths") }}</small>
              <div class="muted">{{ t("eg") }} {{ m.paths[0]?.text }}</div>
            </li>
          </ol>
          <p v-if="!evidence.metapaths.length" class="muted">{{ t("noPath") }}</p>
        </section>
      </div>
    </div>
    <AskPanel @subgraph="showAnswerGraph" @focus="focus = $event" />
  </main>
</template>
