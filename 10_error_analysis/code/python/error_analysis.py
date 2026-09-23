"""Select auditable false-positive and false-negative examples with source metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FINE_TUNED = ("Fine-Tuned CAMeLBERT", "Fine-Tuned MARBERT", "Fine-Tuned Paraphrase")


def labels(values: pd.Series) -> pd.Series:
    """Normalize numeric or textual predictions to the publication label names."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.replace({"0": "non-security", "0.0": "non-security", "1": "security", "1.0": "security"})


def select(frame: pd.DataFrame, dataset: str, rows_per_type: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame[["app_name", "store"]].isna().any().any():
        raise AssertionError("error-analysis inputs require application/store provenance")
    chosen = []
    summaries = []
    for model in FINE_TUNED:
        for error_type, truth, prediction in (
            ("False Positive", "non-security", "security"),
            ("False Negative", "security", "non-security"),
        ):
            candidates = frame.loc[labels(frame["label_name"]).eq(truth) & labels(frame[model]).eq(prediction)].copy()
            candidates = candidates.sort_values(["store", "app_name", "row_id"], kind="stable")
            sample = candidates.head(rows_per_type).copy()
            sample.insert(0, "dataset", dataset)
            sample.insert(1, "model", model)
            sample.insert(2, "error_type", error_type)
            chosen.append(sample)
            summaries.append({
                "dataset": dataset,
                "model": model,
                "error_type": error_type,
                "available_count": len(candidates),
                "selected_count": len(sample),
                "note": "" if len(sample) == rows_per_type else "fewer qualifying errors were available",
            })
    return pd.concat(chosen, ignore_index=True), pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("dataset", choices=["cross_validation", "heldout"])
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    selected, summary = select(pd.read_csv(args.predictions_csv), args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "misclassified_instances.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "sampling_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
