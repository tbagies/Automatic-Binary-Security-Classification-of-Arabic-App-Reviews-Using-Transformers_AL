"""Reproduce the paper's ten-fold SentenceTransformer fine-tuning and 1-NN evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import BatchAllTripletLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


LABEL_TO_ID = {"non-security": 0, "security": 1}
OUTPUT_NAMES = {
    "camelbert": "CAMeLBERT",
    "marbertv2": "MARBERTv2",
    "paraphrase": "paraphrase-multilingual-MiniLM-L12-v2",
}


def predict_one_nearest_neighbor(
    model: SentenceTransformer,
    reference_texts: list[str],
    reference_labels: np.ndarray,
    query_texts: list[str],
) -> np.ndarray:
    """Assign each query the label of its most similar training-fold review."""
    reference_embeddings = model.encode(
        reference_texts, convert_to_tensor=True, show_progress_bar=False
    )
    query_embeddings = model.encode(
        query_texts, convert_to_tensor=True, show_progress_bar=False
    )
    matches = util.semantic_search(query_embeddings, reference_embeddings, top_k=1)
    indices = [match[0]["corpus_id"] for match in matches]
    return np.asarray(reference_labels, dtype=int)[indices]


def metric_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(truth, prediction),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
    }


def run(training_csv: Path, config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(training_csv).reset_index(drop=True)
    required = {"content", "Keyword", "store", "app_name", "review_id"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame["row_id"] = np.arange(len(frame))
    labels = frame["Keyword"].map(LABEL_TO_ID)
    if labels.isna().any():
        raise ValueError("Keyword contains an unsupported label")
    labels = labels.astype(int).to_numpy()
    fine_tuning = config["final_fine_tuning"]
    seed = int(fine_tuning.get("seed", config["seed"]))
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    output.mkdir(parents=True, exist_ok=True)
    metadata = [
        column
        for column in (
            "row_id", "review_id", "content", "Keyword", "store", "app_name",
            "source_file", "source_row_number", "iteration_added"
        )
        if column in frame.columns
    ]

    for model_key, model_name in config["transformers"].items():
        model_output = output / OUTPUT_NAMES[model_key]
        model_output.mkdir(parents=True, exist_ok=True)
        all_predictions = []
        fold_metrics = []
        for fold, (train_index, test_index) in enumerate(
            splitter.split(frame, labels), start=1
        ):
            torch.manual_seed(seed)
            train_frame = frame.iloc[train_index]
            test_frame = frame.iloc[test_index]
            model = SentenceTransformer(model_name)
            model.max_seq_length = min(
                model.max_seq_length, int(fine_tuning.get("max_length", model.max_seq_length))
            )
            train_dataset = Dataset.from_dict({
                "sentence": train_frame["content"].astype(str).tolist(),
                "label": labels[train_index].tolist(),
            })
            temporary = model_output / f"fold_{fold:02d}_training"
            arguments = SentenceTransformerTrainingArguments(
                output_dir=str(temporary),
                num_train_epochs=int(fine_tuning["epochs"]),
                per_device_train_batch_size=int(fine_tuning["batch_size"]),
                warmup_ratio=float(fine_tuning["warmup_ratio"]),
                learning_rate=float(fine_tuning["learning_rate"]),
                lr_scheduler_type=str(fine_tuning["scheduler"]),
                fp16=torch.cuda.is_available(),
                save_strategy="no",
                seed=seed,
                report_to=[],
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=arguments,
                train_dataset=train_dataset,
                loss=BatchAllTripletLoss(model=model),
            )
            trainer.train()
            predicted = predict_one_nearest_neighbor(
                model,
                train_frame["content"].astype(str).tolist(),
                labels[train_index],
                test_frame["content"].astype(str).tolist(),
            )
            truth = labels[test_index]
            fold_metrics.append({
                "model": model_key,
                "fold": fold,
                **metric_row(truth, predicted),
            })
            prediction_frame = test_frame[metadata].copy()
            prediction_frame["fold"] = fold
            prediction_frame["prediction"] = np.where(
                predicted == 1, "security", "non-security"
            )
            all_predictions.append(prediction_frame)
            shutil.rmtree(temporary, ignore_errors=True)

        predictions = pd.concat(all_predictions).sort_values("row_id")
        if len(predictions) != len(frame) or predictions["row_id"].nunique() != len(frame):
            raise AssertionError("each row must have exactly one out-of-fold prediction")
        predictions.to_csv(
            model_output / "oof_predictions.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(fold_metrics).to_csv(
            model_output / "fold_metrics.csv", index=False, encoding="utf-8-sig"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.training_csv, args.config, args.output)


if __name__ == "__main__":
    main()
