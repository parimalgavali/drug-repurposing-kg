"""Precompute frozen sentence embeddings of drug and disease descriptions.

Drug text = description + mechanism of action. The DrugBank `indication` field is excluded:
it names the treated diseases, i.e. the labels.

    uv run --extra ml python -m ml.text_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("artifacts/text_features.npz")
DRUG_FIELDS = ("description", "mechanism_of_action")
DISEASE_FIELDS = ("mondo_name", "mondo_definition", "umls_description")


def texts(df: pl.DataFrame, fields: tuple[str, ...]) -> list[str]:
    joined = pl.concat_str([pl.col(f).fill_null("") for f in fields], separator=" ")
    return df.select(joined.str.strip_chars().alias("t"))["t"].to_list()


def encode(items: list[str], model_name: str = MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(items, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    emb[[not t for t in items]] = 0.0  # no text → zero vector, not the embedding of ""
    return emb.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    drugs = pl.read_parquet(args.processed / "drug_features.parquet").sort("node_index")
    diseases = pl.read_parquet(args.processed / "disease_features.parquet").sort("node_index")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        drug_index=drugs["node_index"].to_numpy(),
        drug_emb=encode(texts(drugs, DRUG_FIELDS)),
        disease_index=diseases["node_index"].to_numpy(),
        disease_emb=encode(texts(diseases, DISEASE_FIELDS)),
    )
    print(f"wrote {args.out}: {drugs.height} drugs, {diseases.height} diseases")


if __name__ == "__main__":
    main()
