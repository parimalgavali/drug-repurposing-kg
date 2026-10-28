<script setup lang="ts">
import { computed } from "vue";
import type { Candidate, Candidates } from "../api/client";
import { nm, t } from "../i18n";

const props = defineProps<{ data: Candidates | null; selected: number | null; loading: boolean }>();
const emit = defineEmits<{ select: [c: Candidate] }>();
const top = computed(() => Math.max(...(props.data?.candidates ?? []).map((c) => c.score), 1e-9));

function trials(c: Candidate): string | null {
  if (c.ct_studies == null) return null;
  return c.ct_studies > 0 ? t("inTrials", c.ct_studies) : t("noTrials");
}
</script>

<template>
  <section class="panel">
    <h2>{{ t("candidates") }}<span v-if="data">{{ ` ${t("for")} ${nm(data.disease.name, data.disease.name_de)}` }}</span></h2>
    <p v-if="loading" class="muted">{{ t("loading") }}</p>
    <p v-else-if="!data" class="muted">{{ t("searchFirst") }}</p>
    <template v-else>
      <p v-if="!data.candidates.length" class="muted">{{ t("noPredictions") }}</p>
      <ol class="candidates">
        <li
          v-for="c in data.candidates"
          :key="c.drug_id"
          data-testid="candidate"
          :class="{ selected: c.drug_id === selected }"
          @click="emit('select', c)"
        >
          <span class="rank">{{ c.rank }}</span>
          <span class="name">{{ nm(c.drug, c.drug_de) }}</span>
          <span class="bar" :title="`score ${c.score}`"><span :style="{ width: `${(100 * Math.max(c.score, 0)) / top}%` }" /></span>
          <span v-if="trials(c)" class="badge" :class="{ none: c.ct_studies === 0 }">{{ trials(c) }}</span>
        </li>
      </ol>
      <details v-if="data.known.length" class="known">
        <summary>{{ t("known") }} ({{ data.known.length }}), {{ t("knownContext") }}</summary>
        <ul>
          <li v-for="k in data.known" :key="`${k.drug_id}-${k.relation}`">
            {{ nm(k.drug, k.drug_de) }} <small>{{ k.relation === "OFF_LABEL_USE" ? t("offLabel") : t("indication") }}</small>
          </li>
        </ul>
      </details>
    </template>
  </section>
</template>
