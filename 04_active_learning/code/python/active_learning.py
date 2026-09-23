"""Five-voter unanimous-agreement Active Learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


VOTER_NAMES = (
    "paraphrase_svm",
    "paraphrase_adaboost",
    "marbert_ridge",
    "camelbert_svm",
    "camelbert_logistic_regression",
)
SOURCE_REVIEW_LIMIT = 10_000
CLASS_CANDIDATE_TARGET = 10
PREDICTION_BATCH_SIZE = 1


class Predictor(Protocol):
    def predict(self, rows: pd.DataFrame) -> np.ndarray: ...


@dataclass
class EmbeddingClassifier:
    encoder: SentenceTransformer
    classifier: object

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        vectors = self.encoder.encode(rows["content"].tolist(), convert_to_numpy=True)
        return np.asarray(self.classifier.predict(vectors), dtype=int)


def load_voters(manifest: Path) -> dict[str, Predictor]:
    config = json.loads(manifest.read_text(encoding="utf-8"))
    encoders: dict[str, SentenceTransformer] = {}
    voters: dict[str, Predictor] = {}
    for name in VOTER_NAMES:
        item = config["voters"][name]
        encoders.setdefault(item["encoder"], SentenceTransformer(item["encoder"]))
        voters[name] = EmbeddingClassifier(encoders[item["encoder"]], joblib.load(item["classifier"]))
    return voters


def stable_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    if "source_row_number" not in ordered:
        ordered["source_row_number"] = ordered.index + 2
    ordered["_source_order"] = pd.to_numeric(ordered["source_row_number"], errors="coerce")
    ordered["_source_order"] = ordered["_source_order"].fillna(np.iinfo(np.int64).max)
    return ordered.sort_values(["_source_order", "review_id"], kind="stable").drop(columns="_source_order")


def source_seed(seed: int, store: str, app_name: str) -> int:
    """Derive a repeatable, independent sampling seed for one app/store source."""
    payload = f"{seed}\x1f{store}\x1f{app_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def select_agreements(
    rows: pd.DataFrame,
    voters: dict[str, Predictor],
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """Scan one source until both unanimous classes reach ten or 10,000 rows."""
    required = {"review_id", "content", "app_name", "store", "source_file", "source_row_number"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    ordered = stable_order(rows).head(SOURCE_REVIEW_LIMIT).copy()
    prediction_columns = [f"pred_{name}" for name in VOTER_NAMES]
    candidate_batches: list[pd.DataFrame] = []
    class_counts = {0: 0, 1: 0}
    rows_processed = 0

    for start in range(0, len(ordered), PREDICTION_BATCH_SIZE):
        batch = ordered.iloc[start : start + PREDICTION_BATCH_SIZE].copy()
        for name, column in zip(VOTER_NAMES, prediction_columns, strict=True):
            batch[column] = voters[name].predict(batch)

        for position in range(len(batch)):
            rows_processed += 1
            candidate = batch.iloc[[position]].copy()
            predictions = candidate[prediction_columns].iloc[0]
            if predictions.nunique() == 1:
                value = int(predictions.iloc[0])
                candidate["proposed_label_numeric"] = value
                candidate_batches.append(candidate)
                class_counts[value] += 1
            if all(class_counts[value] >= CLASS_CANDIDATE_TARGET for value in (0, 1)):
                break
        if all(class_counts[value] >= CLASS_CANDIDATE_TARGET for value in (0, 1)):
            break

    if candidate_batches:
        candidates = pd.concat(candidate_batches, ignore_index=True)
    else:
        candidates = ordered.head(0).copy()
        for column in prediction_columns:
            candidates[column] = pd.Series(dtype="int64")
        candidates["proposed_label_numeric"] = pd.Series(dtype="int64")
    label_name = {0: "non-security", 1: "security"}
    candidates["proposed_label"] = candidates["proposed_label_numeric"].map(label_name)
    store = str(ordered["store"].iloc[0]) if len(ordered) else ""
    app_name = str(ordered["app_name"].iloc[0]) if len(ordered) else ""
    rng_seed = source_seed(seed, store, app_name)
    security_candidates = candidates.loc[
        candidates["proposed_label_numeric"].eq(1)
    ]
    nonsecurity_candidates = candidates.loc[
        candidates["proposed_label_numeric"].eq(0)
    ]
    security_count = min(CLASS_CANDIDATE_TARGET, len(security_candidates))
    nonsecurity_count = min(security_count, len(nonsecurity_candidates))
    selected_parts = [
        security_candidates.sample(
            n=security_count,
            random_state=(rng_seed + 1) % (2**32),
        ),
        nonsecurity_candidates.sample(
            n=nonsecurity_count,
            random_state=rng_seed % (2**32),
        ),
    ]
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = stable_order(selected).reset_index(drop=True)
    selected.attrs["candidate_counts"] = {
        "security": len(security_candidates),
        "non-security": len(nonsecurity_candidates),
    }
    return selected, rows_processed


def safe_name(store: str, app_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{store}__{app_name}").strip("_") or "source"
    digest = hashlib.sha256(f"{store}\x1f{app_name}".encode()).hexdigest()[:8]
    return f"{stem}__{digest}.csv"


def run(pool_csv: Path, voter_manifest: Path, output_dir: Path, seed: int) -> None:
    pool = pd.read_csv(pool_csv)
    voters = load_voters(voter_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for (store, app_name), group in pool.groupby(["store", "app_name"], sort=True):
        selected, rows_processed = select_agreements(group, voters, seed)
        candidate_counts = selected.attrs.get("candidate_counts", {})
        selected.insert(0, "selection_order", range(1, len(selected) + 1))
        selected.to_csv(output_dir / safe_name(str(store), str(app_name)), index=False, encoding="utf-8-sig")
        counts = selected["proposed_label"].value_counts()
        security = int(counts.get("security", 0))
        nonsecurity = int(counts.get("non-security", 0))
        reasons = []
        security_available = int(candidate_counts.get("security", security))
        nonsecurity_available = int(candidate_counts.get("non-security", nonsecurity))
        if security_available < 10:
            reasons.append(
                f"only {security_available} unanimous security rows available before the source stopped"
            )
        if nonsecurity_available < security:
            reasons.append(
                f"only {nonsecurity_available} unanimous non-security rows available to match "
                f"{security} selected security rows"
            )
        elif nonsecurity < nonsecurity_available:
            reasons.append(
                f"randomly limited non-security rows to {nonsecurity} to match the selected security rows"
            )
        summaries.append({
            "store": store,
            "app_name": app_name,
            "rows_processed": rows_processed,
            "security_candidates": security_available,
            "nonsecurity_candidates": nonsecurity_available,
            "security_rows": security,
            "nonsecurity_rows": nonsecurity,
            "rows_written": len(selected),
            "shortfall_reason": "; ".join(reasons),
        })
    pd.DataFrame(summaries).to_csv(output_dir.parent / "agreement_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool_csv", type=Path)
    parser.add_argument("voter_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    run(args.pool_csv, args.voter_manifest, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
