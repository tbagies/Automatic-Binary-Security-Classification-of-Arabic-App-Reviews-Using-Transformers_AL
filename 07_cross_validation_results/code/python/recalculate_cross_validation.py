"""Recalculate 12-pipeline cross-validation metrics from authoritative OOF predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


META_COLUMNS = {
    "row_id", "training_row_id", "review_id", "content", "Keyword", "store", "app_name",
    "label_source", "iteration_added", "source_row_number", "review_status", "validation_id",
    "proposed_label", "source_file", "candidate_pool_reason", "manual_final_label", "manual_status",
    "final_category", "manual_validation_reason", "manual_label_changed", "selected_for_proposed_addition",
    "selection_balance_origin", "approval_status", "text", "label_name", "label_id", "fold",
}


def labels(values: pd.Series) -> pd.Series:
    """Normalize numeric or textual predictions to the publication label names."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.replace({"0": "non-security", "0.0": "non-security", "1": "security", "1.0": "security"})


def score(truth: pd.Series, prediction: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(truth, prediction),
        "precision": precision_score(truth, prediction, average="macro", zero_division=0),
        "recall": recall_score(truth, prediction, average="macro", zero_division=0),
        "f1": f1_score(truth, prediction, average="macro"),
    }


def run(predictions_csv: Path, output: Path) -> None:
    frame = pd.read_csv(predictions_csv)
    if frame[["app_name", "store"]].isna().any().any():
        raise AssertionError("review-level predictions require application/store provenance")
    pipelines = [column for column in frame.columns if column not in META_COLUMNS]
    if len(pipelines) != 12:
        raise AssertionError(f"expected 12 pipelines, found {len(pipelines)}")
    rows = []
    for fold, fold_frame in frame.groupby("fold", sort=True):
        for pipeline in pipelines:
            rows.append({"fold": fold, "pipeline": pipeline, **score(labels(fold_frame["label_name"]), labels(fold_frame[pipeline]))})
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("pipeline", sort=False)[["accuracy", "precision", "recall", "f1"]].mean().reset_index()
    summary = summary.rename(columns={column: f"mean_{column}" for column in ("accuracy", "precision", "recall", "f1")})
    summary.insert(1, "fold_count", 10)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "cross_validation_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "cross_validation_summary_metrics.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.predictions_csv, args.output)


if __name__ == "__main__":
    main()
