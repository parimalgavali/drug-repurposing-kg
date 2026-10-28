import { expect, test } from "@playwright/test";

// Scripted walk-through for the README GIF: search → candidate (trial badge, evidence graph)
// → question with citations → DE toggle and a German search.
test("demo", async ({ page }) => {
  const pause = (ms = 1200) => page.waitForTimeout(ms);
  await page.goto("/");
  await pause();

  await page.getByLabel("Search a disease").pressSequentially("asthma", { delay: 90 });
  await page.getByRole("option", { name: /^asthma/ }).click();
  const first = page.getByTestId("candidate").first();
  await expect(first).toBeVisible();
  await pause();
  await first.click();
  await expect(first.getByTestId("trials")).toHaveText(/trial/, { timeout: 30_000 });
  await expect(page.getByTestId("evidence")).toContainText("paths found", { timeout: 30_000 });
  await pause(2500);

  await page.getByLabel("Question").pressSequentially("Which genes does Ibuprofen target?", {
    delay: 50,
  });
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  const answer = page.getByTestId("answer");
  await expect(answer).toContainText("[4425]", { timeout: 90_000 });
  await answer.scrollIntoViewIfNeeded();
  await pause(2500);
  await page.getByTestId("cite").filter({ hasText: "4425" }).first().click();
  await page.locator("h1").scrollIntoViewIfNeeded();
  await pause(2500);

  await page.getByRole("button", { name: "DE", exact: true }).click();
  await pause();
  const search = page.getByLabel("Krankheit suchen");
  await search.fill("");
  await search.pressSequentially("Skorbut", { delay: 110 });
  await page.getByRole("option", { name: /^Skorbut/ }).first().click();
  await expect(page.getByTestId("candidate").first()).toBeVisible();
  await pause(2500);
});
