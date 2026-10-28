import { expect, test } from "@playwright/test";

// Runs against the fixture graph (scripts/demo_fixture.py) with LLM_PROVIDER=recorded and
// LLM_RECORDINGS=web/e2e/recordings. Re-record with LLM_PROVIDER=anthropic LLM_RECORD=1.
// ClinicalTrials.gov lookups are stubbed, so CI never calls CT.gov.
const TRIALS = {
  studies: 12,
  max_phase: 3,
  statuses: { COMPLETED: 12 },
  url: "https://clinicaltrials.gov/search?cond=asthma&intr=x",
  cached: false,
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/pairs/*/*/trials", (route) => route.fulfill({ json: TRIALS }));
});

test("search → candidate → graph → ask", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("note")).toContainText("not medical advice");

  await page.getByLabel("Search a disease").fill("asthma");
  await page.getByRole("option", { name: /^asthma/ }).click();
  const first = page.getByTestId("candidate").first();
  await expect(first).toBeVisible();
  await first.click();
  await expect(page.getByTestId("evidence")).toContainText("paths found");
  await expect(page.getByTestId("graph")).not.toHaveAttribute("data-nodes", "0");

  await page.getByLabel("Question").fill("Which genes does Ibuprofen target?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(page.getByTestId("answer")).toContainText("[4425]", { timeout: 30_000 });
  await page.getByText("Show Cypher").click();
  await expect(page.getByTestId("cypher").first()).toContainText("MATCH");
  await page.getByTestId("cite").filter({ hasText: "4425" }).first().click();
  await expect(page.getByTestId("graph")).not.toHaveAttribute("data-nodes", "0");
});

test("DE toggle shows German UI text and Wikidata names", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "DE", exact: true }).click();
  await expect(page.getByRole("note")).toContainText("keine medizinische Beratung");
  await page.getByLabel("Krankheit suchen").fill("asthma");
  await page.getByRole("option", { name: /^Asthma/ }).first().click();
  await expect(page.getByRole("heading", { name: /Kandidaten-Medikamente für Asthma/ })).toBeVisible();
  await expect(page.getByTestId("candidate").first()).toBeVisible();
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(page.getByRole("heading", { name: /Candidate drugs for asthma/ })).toBeVisible();
});

test("clicking a candidate shows its ClinicalTrials.gov badge", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Search a disease").fill("asthma");
  await page.getByRole("option", { name: /^asthma/ }).click();
  const first = page.getByTestId("candidate").first();
  await first.click();
  const badge = first.getByTestId("trials");
  await expect(badge).toHaveText("in 12 trials");
  await expect(badge).toHaveAttribute("href", /^https:\/\/clinicaltrials\.gov\/search\?/);
  await expect(badge).toHaveAttribute("title", /highest phase 3/);
});

test("search finds a disease by its German name", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Search a disease").fill("Skorbut");
  await page.getByRole("option", { name: /^scurvy/ }).click();
  await expect(page.getByTestId("candidate").first()).toBeVisible();
});
