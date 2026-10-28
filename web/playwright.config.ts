import { defineConfig } from "@playwright/test";

// Expects the API on :8000 (fixture graph + recorded LLM, see .github/workflows/ci.yml).
export default defineConfig({
  testDir: "e2e",
  testMatch: "*.e2e.ts",
  timeout: 60_000,
  use: { baseURL: process.env.WEB_URL ?? "http://localhost:4173", trace: "retain-on-failure" },
  webServer: process.env.WEB_URL
    ? undefined
    : {
        command: "npm run build && npm run preview -- --port 4173 --strictPort",
        url: "http://localhost:4173",
        reuseExistingServer: true,
        timeout: 120_000,
      },
});
