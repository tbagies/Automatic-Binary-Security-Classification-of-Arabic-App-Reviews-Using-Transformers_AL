"""Build the paper-described negative/neutral review pool with provenance."""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd


NON_ARABIC = re.compile(r"[^\u0600-\u06FF\s]")


def normalize_text(value: object) -> str:
    """Remove non-Arabic tokens/symbols and normalize whitespace.

    Diacritics, stopwords, repeated tokens, and morphology-invalid tokens are not
    removed here because those stricter operations are absent from the paper's
    preprocessing description.
    """
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = NON_ARABIC.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_review_id(store: str, app_name: str, text: str) -> str:
    payload = "\x1f".join((store, app_name, normalize_text(text))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preprocess(input_csv: Path, output_csv: Path) -> None:
    frame = pd.read_csv(input_csv)
    required = {
        "content",
        "app_name",
        "store",
        "sentiment_camel",
        "sentiment_arabert",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing provenance columns: {sorted(missing)}")
    frame["sentiment_camel"] = frame["sentiment_camel"].astype("string").str.lower().str.strip()
    frame["sentiment_arabert"] = frame["sentiment_arabert"].astype("string").str.lower().str.strip()
    frame = frame.loc[
        frame["sentiment_camel"].eq(frame["sentiment_arabert"])
        & frame["sentiment_camel"].isin({"negative", "neutral"})
    ].copy()
    frame["content"] = frame["content"].map(normalize_text)
    frame = frame.loc[frame["content"].ne("")].copy()
    frame["source_file"] = frame.get("source_file", input_csv.name)
    frame["source_row_number"] = frame.get(
        "source_row_number", pd.Series(frame.index + 2, index=frame.index)
    )
    frame["review_id"] = [
        stable_review_id(store, app, text)
        for store, app, text in frame[["store", "app_name", "content"]].itertuples(index=False)
    ]
    frame = frame.drop_duplicates("content", keep="first")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    preprocess(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
