FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra graph --extra llm --extra api
COPY api ./api
COPY explain ./explain
COPY graphrag ./graphrag
COPY kg ./kg
COPY llm ./llm
EXPOSE 8000
USER nobody
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
