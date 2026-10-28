import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("parses events split across chunks", () => {
    const p = new SseParser();
    expect(p.push('event: token\ndata: {"text": "Ibu')).toEqual([]);
    expect(p.push('profen [14287]"}\n\nevent: done\ndata: {"intent": "neighbours"}\n\n')).toEqual([
      { event: "token", data: { text: "Ibuprofen [14287]" } },
      { event: "done", data: { intent: "neighbours" } },
    ]);
  });

  it("handles CRLF line endings", () => {
    expect(new SseParser().push('event: rows\r\ndata: {"total": 2}\r\n\r\n')).toEqual([
      { event: "rows", data: { total: 2 } },
    ]);
  });
});
