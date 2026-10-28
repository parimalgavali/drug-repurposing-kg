<script setup lang="ts">
import { ref } from "vue";
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

const list = ref<Candidates | null>(null);
const loading = ref(false);
const selected = ref<number | null>(null);
const evidence = ref<Explanation | null>(null);
const graph = ref<Cytoscape | null>(null);
const title = ref("Evidence graph");
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
    title.value = `Why ${c.drug} → ${list.value.disease.name}?`;
    focus.value = null;
  } catch (e) {
    error.value = String(e);
  }
}

function showAnswerGraph(g: Cytoscape) {
  graph.value = g;
  title.value = "Answer subgraph";
  evidence.value = null;
}
</script>

<template>
  <header class="banner" role="note">
    <strong>Research demo, not medical advice.</strong> Predictions are hypotheses from a graph model,
    not evidence that a drug is safe or effective.
  </header>
  <main>
    <div class="intro">
      <h1>RepurposeKG</h1>
      <p class="muted">Drug-repurposing hypotheses from PrimeKG, explained with graph paths.</p>
      <DiseaseSearch @pick="pickDisease" />
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </div>
    <div class="grid">
      <CandidateList :data="list" :selected="selected" :loading="loading" @select="pickCandidate" />
      <div>
        <GraphView :graph="graph" :focus="focus" :title="title" @focus="focus = $event" />
        <section v-if="evidence" class="panel" data-testid="evidence">
          <h3>{{ evidence.paths_found }} paths found{{ evidence.truncated ? " (truncated)" : "" }}</h3>
          <ol>
            <li v-for="m in evidence.metapaths" :key="m.metapath">
              <code>{{ m.metapath }}</code> <small>DWPC {{ m.dwpc.toPrecision(3) }} · {{ m.count }} paths</small>
              <div class="muted">e.g. {{ m.paths[0]?.text }}</div>
            </li>
          </ol>
          <p v-if="!evidence.metapaths.length" class="muted">No 2–3 hop mechanistic path found.</p>
        </section>
      </div>
    </div>
    <AskPanel @subgraph="showAnswerGraph" @focus="focus = $event" />
  </main>
</template>
