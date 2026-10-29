import { afterEach, expect, test, vi } from "vitest";
import { ask, search } from "./static";

afterEach(() => vi.unstubAllGlobals());

function serve(files: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const key = Object.keys(files).find((k) => url.endsWith(k));
      return key ? new Response(JSON.stringify(files[key])) : new Response("", { status: 404 });
    }),
  );
}

test("search ranks exact, prefix and German matches in the browser", async () => {
  serve({
    "search.json": [
      [1, "allergic asthma", null],
      [2, "asthma", "Asthma"],
      [3, "scurvy", "Skorbut"],
      [4, "asthma, occupational", null],
    ],
  });
  expect((await search("asthma", 8)).map((h) => h.id)).toEqual([2, 4, 1]);
  expect((await search("skorbut", 8)).map((h) => [h.id, h.label])).toEqual([[3, "Disease"]]);
});

test("ask replays saved answers and refuses other questions", async () => {
  const events = [{ event: "token", data: { text: "PTGS2 [4425]" } }, { event: "done", data: {} }];
  serve({ "answers/index.json": { en: { "Which genes?": "0-en.json" } }, "answers/0-en.json": events });
  const seen: unknown[] = [];
  await ask("Which genes?", "en", (e) => seen.push(e));
  expect(seen).toEqual(events);
  await expect(ask("Something else?", "en", () => {})).rejects.toThrow(/example questions/);
});
