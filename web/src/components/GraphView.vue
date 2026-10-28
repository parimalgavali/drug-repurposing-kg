<script setup lang="ts">
import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { Cytoscape } from "../api/client";
import { color } from "../colors";

const props = defineProps<{ graph: Cytoscape | null; focus: string | null; title: string }>();
const emit = defineEmits<{ focus: [id: string] }>();
const el = ref<HTMLDivElement>();
let cy: Core | null = null;

const types = computed(() => [...new Set((props.graph?.nodes ?? []).map((n) => n.data.type ?? ""))]);

function render() {
  cy?.destroy();
  cy = null;
  if (!el.value || !props.graph?.nodes.length) return;
  const nodes: ElementDefinition[] = props.graph.nodes.map((n) => ({
    data: { ...n.data, color: color(n.data.type ?? "") },
  }));
  const edges: ElementDefinition[] = props.graph.edges.map((e) => ({ data: { ...e.data } }));
  cy = cytoscape({
    container: el.value,
    elements: [...nodes, ...edges],
    layout: { name: "cose", animate: false, padding: 20 },
    style: [
      {
        selector: "node",
        style: {
          "background-color": "data(color)",
          label: "data(label)",
          "font-size": 10,
          "text-valign": "bottom",
          "text-margin-y": 4,
          width: 18,
          height: 18,
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.5,
          "line-color": "#cbd5e1",
          label: "data(label)",
          "font-size": 7,
          color: "#64748b",
          "curve-style": "bezier",
          "text-rotation": "autorotate",
        },
      },
      { selector: ".focus", style: { "border-width": 4, "border-color": "#facc15", width: 28, height: 28 } },
    ],
  });
  cy.on("tap", "node", (e) => emit("focus", e.target.id()));
  applyFocus();
}

function applyFocus() {
  if (!cy) return;
  cy.nodes().removeClass("focus");
  if (!props.focus) return;
  const node = cy.getElementById(props.focus);
  if (node.nonempty()) {
    node.addClass("focus");
    cy.animate({ center: { eles: node } }, { duration: 300 });
  }
}

watch(() => props.graph, render);
watch(() => props.focus, applyFocus);
onMounted(render);
onBeforeUnmount(() => cy?.destroy());
</script>

<template>
  <section class="panel graph">
    <h2>{{ title }}</h2>
    <p v-if="!graph?.nodes.length" class="muted">Pick a candidate or ask a question to see its subgraph.</p>
    <div ref="el" class="cy" data-testid="graph" :data-nodes="graph?.nodes.length ?? 0" />
    <ul class="legend">
      <li v-for="t in types" :key="t"><span class="dot" :style="{ background: color(t) }" />{{ t }}</li>
    </ul>
  </section>
</template>
