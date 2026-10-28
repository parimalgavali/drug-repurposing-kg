"""Write the OpenAPI schema for the frontend types: `make openapi`."""

from __future__ import annotations

import json
from pathlib import Path

from api.main import app

OUT = Path("docs/openapi.json")


def schema() -> str:
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.write_text(schema())
    print(f"wrote {OUT}")
