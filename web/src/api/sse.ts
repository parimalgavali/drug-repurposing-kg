export interface SseEvent {
  event: string;
  data: unknown;
}

/** Incremental parser for `text/event-stream` chunks (single-line JSON `data:` fields). */
export class SseParser {
  private buf = "";

  push(chunk: string): SseEvent[] {
    this.buf += chunk.replace(/\r\n/g, "\n");
    const out: SseEvent[] = [];
    let end: number;
    while ((end = this.buf.indexOf("\n\n")) >= 0) {
      const block = this.buf.slice(0, end);
      this.buf = this.buf.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) out.push({ event, data: JSON.parse(data.join("\n")) });
    }
    return out;
  }
}
