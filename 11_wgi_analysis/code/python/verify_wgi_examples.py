"""Verify the qualitative WGI examples recorded for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def verify(examples_csv: Path) -> dict[str, int]:
    frame = pd.read_csv(examples_csv)
    required = {"example_id", "source_kind", "review", "predicted_label", "app_name", "store", "explanation_ar"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[["app_name", "store", "review", "predicted_label", "explanation_ar"]].isna().any().any():
        raise AssertionError("WGI examples contain incomplete review metadata")
    if frame["example_id"].duplicated().any() or frame["review"].duplicated().any():
        raise AssertionError("WGI examples must be unique")
    return {
        "examples": len(frame),
        "security": int(frame["predicted_label"].eq("security").sum()),
        "nonsecurity": int(frame["predicted_label"].eq("non-security").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("examples_csv", type=Path)
    args = parser.parse_args()
    print(verify(args.examples_csv))


if __name__ == "__main__":
    main()

