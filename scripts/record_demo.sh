#!/usr/bin/env bash
# Re-record docs/img/demo.gif from the scripted walk-through (web/e2e/demo.record.ts).
# Needs the full-graph API with live Claude on $API_URL (default http://localhost:8001), e.g.
#   LLM_PROVIDER=anthropic uv run --extra graph --extra llm --extra api --extra rdf \
#     uvicorn api.main:app --port 8001
set -euo pipefail
cd "$(dirname "$0")/../web"
rm -rf test-results/demo
npx playwright test --config playwright.demo.config.ts
video=$(find test-results/demo -name '*.webm' | head -1)
ffmpeg -loglevel error -y -i "$video" \
  -vf "fps=6,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer" \
  ../docs/img/demo.gif
ls -l ../docs/img/demo.gif
