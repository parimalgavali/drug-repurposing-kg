<script setup lang="ts">
import { ref, watch } from "vue";
import { type SearchHit, search } from "../api/client";

const emit = defineEmits<{ pick: [hit: SearchHit] }>();
const q = ref("");
const hits = ref<SearchHit[]>([]);
const open = ref(false);
let timer: ReturnType<typeof setTimeout> | undefined;
let latest = 0;

watch(q, (value) => {
  clearTimeout(timer);
  if (value.trim().length < 2) {
    hits.value = [];
    return;
  }
  timer = setTimeout(async () => {
    const id = ++latest;
    const found = await search(value.trim()).catch(() => []);
    if (id === latest) {
      hits.value = found;
      open.value = true;
    }
  }, 200);
});

function pick(hit: SearchHit) {
  q.value = hit.name;
  open.value = false;
  emit("pick", hit);
}
</script>

<template>
  <div class="search">
    <input
      v-model="q"
      type="search"
      placeholder="Search a disease, e.g. asthma"
      aria-label="Search a disease"
      @focus="open = hits.length > 0"
      @keydown.enter="hits[0] && pick(hits[0])"
    />
    <ul v-if="open && hits.length" class="suggestions" role="listbox">
      <li v-for="h in hits" :key="h.id" role="option" @mousedown.prevent="pick(h)">
        {{ h.name }} <small>[{{ h.id }}]</small>
      </li>
    </ul>
  </div>
</template>
