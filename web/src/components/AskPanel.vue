<script setup lang="ts">
import { computed, ref } from "vue";
import { type AskEvents, type Cytoscape, type Entity, ask } from "../api/client";

const emit = defineEmits<{ subgraph: [g: Cytoscape]; focus: [id: string] }>();
const EXAMPLES = [
  "Which genes does Ibuprofen target?",
  "Which drugs could be repurposed for asthma?",
  "Why might Salbutamol work for asthma?",
];
const question = ref("");
const text = ref("");
const cypher = ref<string[]>([]);
const entities = ref<Entity[]>([]);
const rows = ref(0);
const error = ref<string | null>(null);
const busy = ref(false);

const parts = computed(() =>
  text.value
    .replace(/\*\*/g, "")
    .split(/(\[\d+\])/)
    .map((p) => {
      const m = /^\[(\d+)\]$/.exec(p);
      return m ? { cite: m[1] } : { text: p };
    }),
);

async function submit(q = question.value) {
  if (busy.value || q.trim().length < 3) return;
  question.value = q;
  text.value = "";
  cypher.value = [];
  entities.value = [];
  rows.value = 0;
  error.value = null;
  busy.value = true;
  try {
    await ask(q.trim(), ({ event, data }) => {
      if (event === "entities") entities.value = data as AskEvents["entities"];
      else if (event === "cypher") cypher.value = (data as AskEvents["cypher"]).queries;
      else if (event === "rows") rows.value = (data as AskEvents["rows"]).total;
      else if (event === "token") text.value += (data as AskEvents["token"]).text;
      else if (event === "subgraph") emit("subgraph", data as AskEvents["subgraph"]);
      else if (event === "error") error.value = (data as AskEvents["error"]).message;
    });
  } catch (e) {
    error.value = String(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <section class="panel ask">
    <h2>Ask the graph</h2>
    <form @submit.prevent="submit()">
      <input v-model="question" placeholder="Ask a question in English" aria-label="Question" />
      <button type="submit" :disabled="busy">{{ busy ? "Thinking…" : "Ask" }}</button>
    </form>
    <p class="examples">
      Try:
      <button v-for="ex in EXAMPLES" :key="ex" type="button" class="link" @click="submit(ex)">{{ ex }}</button>
    </p>
    <p v-if="entities.length" class="muted">
      Linked:
      <span v-for="e in entities" :key="e.slot">{{ e.name ?? e.mention }}<small v-if="e.id"> [{{ e.id }}]</small> </span>
      <span v-if="rows"> · {{ rows }} rows</span>
    </p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <div v-if="text" class="answer" data-testid="answer">
      <template v-for="(p, i) in parts" :key="i">
        <button v-if="p.cite" type="button" class="cite" data-testid="cite" @click="emit('focus', p.cite)">
          [{{ p.cite }}]
        </button>
        <span v-else>{{ p.text }}</span>
      </template>
    </div>
    <details v-if="cypher.length" class="cypher">
      <summary>Show Cypher</summary>
      <pre v-for="(c, i) in cypher" :key="i" data-testid="cypher">{{ c }}</pre>
    </details>
  </section>
</template>
