<script setup lang="ts">
import { computed } from "vue";
import type { Candidate, Candidates, TrialState } from "../api/client";
import { nm, t } from "../i18n";

const props = defineProps<{
  data: Candidates | null;
  selected: number | null;
  loading: boolean;
  trials: Record<number, TrialState>;
}>();
const emit = defineEmits<{ select: [c: Candidate] }>();
const top = computed(() => Math.max(...(props.data?.candidates ?? []).map((c) => c.score), 1e-9));

interface Badge {
  text: string;
  none: boolean;
  url?: string;
  title?: string;
}

function badge(c: Candidate): Badge | null {
  const s = props.trials[c.drug_id];
  if (s === "loading") return { text: t("checkingTrials"), none: true };
  const found = typeof s === "object" ? s : null;
  const n = found ? found.studies : c.ct_studies;
  if (n == null) return s === "error" ? { text: t("trialsFailed"), none: true } : null;
  const phase = found ? found.max_phase : c.ct_max_phase;
  return {
    text: n > 0 ? t("inTrials", n) : t("noTrials"),
    none: n === 0,
    url: found?.url,
    title: found && phase != null ? t("maxPhase", phase) : undefined,
  };
}

const badges = computed(() => new Map((props.data?.candidates ?? []).map((c) => [c.drug_id, badge(c)])));
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
          <component
            :is="badges.get(c.drug_id)?.url ? 'a' : 'span'"
            v-if="badges.get(c.drug_id)"
            class="badge"
            data-testid="trials"
            :class="{ none: badges.get(c.drug_id)?.none }"
            :href="badges.get(c.drug_id)?.url"
            :title="badges.get(c.drug_id)?.title"
            target="_blank"
            rel="noopener"
            @click.stop
            >{{ badges.get(c.drug_id)?.text }}</component
          >
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
