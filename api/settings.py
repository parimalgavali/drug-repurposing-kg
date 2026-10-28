"""API settings from the environment (or .env)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    llm_provider: str = "recorded"
    rdf_path: str = "data/rdf/export.ttl"
    ctgov_cache: str = "artifacts/ctgov_cache.jsonl"
    cors_origins: list[str] = ["http://localhost:5173"]
