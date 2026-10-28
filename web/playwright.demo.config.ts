import { defineConfig } from "@playwright/test";

// Records docs/img/demo.gif (see scripts/record_demo.sh). Expects the full-graph API on
// API_URL (default :8001) with LLM_PROVIDER=anthropic; not part of CI.
const api = process.env.API_URL ?? "http://localhost:8001";
const size = { width: 1280, height: 800 };

export default defineConfig({
  testDir: "e2e",
  testMatch: "demo.record.ts",
  outputDir: "test-results/demo",
  timeout: 180_000,
  use: { baseURL: "http://localhost:4174", viewport: size, video: { mode: "on", size } },
  webServer: {
    command: `npm run build && API_URL=${api} npm run preview -- --port 4174 --strictPort`,
    url: "http://localhost:4174",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
