<script setup lang="ts">
import { computed } from "vue";
import type { Candidate, Candidates } from "../api/client";

const props = defineProps<{ data: Candidates | null; selected: number | null; loading: boolean }>();
const emit = defineEmits<{ select: [c: Candidate] }>();
const top = computed(() => Math.max(...(props.data?.candidates ?? []).map((c) => c.score), 1e-9));

function trials(c: Candidate): string | null {
  if (c.ct_studies == null) return null;
  return c.ct_studies > 0 ? `in ${c.ct_studies} trial${c.ct_studies === 1 ? "" : "s"}` : "no trials found";
}
</script>

<template>
  <section class="panel">
    <h2>Candidate drugs<span v-if="data"> for {{ data.disease.name }}</span></h2>
    <p v-if="loading" class="muted">Loading…</p>
    <p v-else-if="!data" class="muted">Search a disease to see model-predicted candidates.</p>
    <template v-else>
      <p v-if="!data.candidates.length" class="muted">No predictions stored for this disease.</p>
      <ol class="candidates">
        <li
          v-for="c in data.candidates"
          :key="c.drug_id"
          data-testid="candidate"
          :class="{ selected: c.drug_id === selected }"
          @click="emit('select', c)"
        >
          <span class="rank">{{ c.rank }}</span>
          <span class="name">{{ c.drug }}</span>
          <span class="bar" :title="`score ${c.score}`"><span :style="{ width: `${(100 * Math.max(c.score, 0)) / top}%` }" /></span>
          <span v-if="trials(c)" class="badge" :class="{ none: c.ct_studies === 0 }">{{ trials(c) }}</span>
        </li>
      </ol>
      <details v-if="data.known.length" class="known">
        <summary>Known treatments in the graph ({{ data.known.length }}), shown for context</summary>
        <ul>
          <li v-for="k in data.known" :key="`${k.drug_id}-${k.relation}`">
            {{ k.drug }} <small>{{ k.relation === "OFF_LABEL_USE" ? "off-label" : "indication" }}</small>
          </li>
        </ul>
      </details>
    </template>
  </section>
</template>
