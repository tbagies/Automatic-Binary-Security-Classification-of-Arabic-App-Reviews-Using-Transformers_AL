"""Pairwise McNemar and paired tests for cross-validation and heldout predictions."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, ttest_rel
from sklearn.metrics import f1_score


META = {
    "row_id", "training_row_id", "review_id", "content", "Keyword", "store", "app_name",
    "label_source", "iteration_added", "source_row_number", "review_status", "validation_id",
    "proposed_label", "source_file", "candidate_pool_reason", "manual_final_label", "manual_status",
    "final_category", "manual_validation_reason", "manual_label_changed", "selected_for_proposed_addition",
    "selection_balance_origin", "approval_status", "text", "label_name", "label_id", "fold", "review", "true_label",
}


def labels(values: pd.Series) -> pd.Series:
    """Normalize numeric or textual predictions to the publication label names."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.replace({"0": "non-security", "0.0": "non-security", "1": "security", "1.0": "security"})


def mcnemar(first: np.ndarray, second: np.ndarray) -> float:
    b = int(np.sum(first & ~second))
    c = int(np.sum(~first & second))
    discordant = b + c
    if discordant == 0:
        return 1.0
    if discordant < 25:
        return float(binomtest(min(b, c), discordant, 0.5, alternative="two-sided").pvalue)
    statistic = (abs(b - c) - 1) ** 2 / discordant
    return float(chi2.sf(statistic, 1))


def pipeline_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META]


def paired_pvalue(first: np.ndarray, second: np.ndarray) -> float:
    """Return a defined p-value for identical vectors and a paired t-test otherwise."""
    if np.array_equal(first, second):
        return 1.0
    value = float(ttest_rel(first, second).pvalue)
    return 0.0 if np.isnan(value) else value


def run(cv_csv: Path, heldout_csv: Path, output: Path) -> None:
    cv = pd.read_csv(cv_csv)
    heldout = pd.read_csv(heldout_csv)
    pipelines = pipeline_columns(cv)
    if pipelines != pipeline_columns(heldout) or len(pipelines) != 12:
        raise AssertionError("cross-validation and heldout pipeline columns must match")
    output.mkdir(parents=True, exist_ok=True)

    for name, frame in (("cross_validation", cv), ("heldout", heldout)):
        truth = labels(frame["label_name"])
        correct = {pipeline: labels(frame[pipeline]).eq(truth).to_numpy() for pipeline in pipelines}
        matrix = pd.DataFrame(index=pipelines, columns=pipelines, dtype=float)
        for first, second in combinations(pipelines, 2):
            matrix.loc[first, second] = matrix.loc[second, first] = mcnemar(correct[first], correct[second])
        matrix.to_csv(output / f"{name}_mcnemar_pvalues.csv", encoding="utf-8-sig")

    fold_accuracy = {
        pipeline: cv.assign(correct=labels(cv[pipeline]).eq(labels(cv["label_name"]))).groupby("fold")["correct"].mean().to_numpy()
        for pipeline in pipelines
    }
    fold_f1 = {
        pipeline: np.asarray([
            f1_score(labels(group["label_name"]), labels(group[pipeline]), average="macro")
            for _, group in cv.groupby("fold", sort=True)
        ])
        for pipeline in pipelines
    }
    for metric_name, values in (("accuracy", fold_accuracy), ("f1", fold_f1)):
        matrix = pd.DataFrame(index=pipelines, columns=pipelines, dtype=float)
        for first, second in combinations(pipelines, 2):
            value = paired_pvalue(values[first], values[second])
            matrix.loc[first, second] = matrix.loc[second, first] = value
        matrix.to_csv(output / f"cross_validation_paired_{metric_name}_pvalues.csv", encoding="utf-8-sig")

    heldout_correct = {
        pipeline: labels(heldout[pipeline]).eq(labels(heldout["label_name"])).astype(float).to_numpy()
        for pipeline in pipelines
    }
    matrix = pd.DataFrame(index=pipelines, columns=pipelines, dtype=float)
    for first, second in combinations(pipelines, 2):
        value = paired_pvalue(heldout_correct[first], heldout_correct[second])
        matrix.loc[first, second] = matrix.loc[second, first] = value
    matrix.to_csv(output / "heldout_paired_accuracy_pvalues.csv", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cv_csv", type=Path)
    parser.add_argument("heldout_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cv_csv, args.heldout_csv, args.output)


if __name__ == "__main__":
    main()
