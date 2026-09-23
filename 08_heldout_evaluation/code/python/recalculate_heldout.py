"""Recalculate heldout metrics from the authoritative 12-pipeline prediction table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


META_COLUMNS = {
    "row_id", "store", "app_name", "review", "true_label", "text", "label_name", "label_id",
    "source_file", "source_row_number", "review_id",
}


def labels(values: pd.Series) -> pd.Series:
    """Normalize numeric or textual predictions to the publication label names."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.replace({"0": "non-security", "0.0": "non-security", "1": "security", "1.0": "security"})


def run(predictions_csv: Path, output_csv: Path) -> None:
    frame = pd.read_csv(predictions_csv)
    if len(frame) != 1_000:
        raise AssertionError("heldout prediction table must contain 1,000 rows")
    if frame[["app_name", "store"]].isna().any().any():
        raise AssertionError("heldout predictions require application/store provenance")
    pipelines = [column for column in frame.columns if column not in META_COLUMNS]
    if len(pipelines) != 12:
        raise AssertionError(f"expected 12 pipelines, found {len(pipelines)}")
    rows = []
    for pipeline in pipelines:
        truth = labels(frame["label_name"])
        prediction = labels(frame[pipeline])
        rows.append({
            "pipeline": pipeline,
            "accuracy": accuracy_score(truth, prediction),
            "precision": precision_score(truth, prediction, average="macro", zero_division=0),
            "recall": recall_score(truth, prediction, average="macro", zero_division=0),
            "f1": f1_score(truth, prediction, average="macro"),
        })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    run(args.predictions_csv, args.output_csv)


if __name__ == "__main__":
    main()
