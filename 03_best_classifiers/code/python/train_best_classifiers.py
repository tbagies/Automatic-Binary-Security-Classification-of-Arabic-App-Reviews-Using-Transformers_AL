"""Train the five LoRA-representation voters selected for Active Learning."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import BatchHardTripletLoss
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


MODEL_KEYS = ("paraphrase", "marbertv2", "camelbert")


def attach_lora(model: SentenceTransformer, lora: dict) -> SentenceTransformer:
    config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=lora["bias"],
        target_modules=list(lora["target_modules"]),
        task_type="FEATURE_EXTRACTION",
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, config)
    return model


def train_encoder(model_name: str, texts: list[str], labels: np.ndarray, config: dict, output: Path) -> SentenceTransformer:
    model = attach_lora(SentenceTransformer(model_name), config["lora"])
    dataset = Dataset.from_dict({"sentence": texts, "label": labels.tolist()})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output),
        num_train_epochs=int(config["active_learning_fine_tuning"]["epochs"]),
        learning_rate=float(config["active_learning_fine_tuning"]["learning_rate"]),
        warmup_ratio=float(config["active_learning_fine_tuning"]["warmup_ratio"]),
        per_device_train_batch_size=int(config["active_learning_fine_tuning"]["batch_size"]),
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        seed=int(config["seed"]),
        save_strategy="no",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=BatchHardTripletLoss(model=model),
    )
    trainer.train()
    model.save(str(output / "adapter_encoder"))
    return model


def voter_specs(seed: int) -> dict[str, tuple[str, object]]:
    return {
        "paraphrase_svm": ("paraphrase", SVC(kernel="rbf")),
        "paraphrase_adaboost": ("paraphrase", AdaBoostClassifier(random_state=seed)),
        "marbert_ridge": ("marbertv2", RidgeClassifier()),
        "camelbert_svm": ("camelbert", SVC(kernel="rbf")),
        "camelbert_logistic_regression": ("camelbert", LogisticRegression(max_iter=4000, random_state=seed)),
    }


def run(training_csv: Path, config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(training_csv)
    required = {"content", "Keyword", "app_name", "store"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    texts = frame["content"].astype(str).tolist()
    labels = frame["Keyword"].map({"non-security": 0, "security": 1}).to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    encoders = {
        key: train_encoder(config["transformers"][key], texts, labels, config, output / key)
        for key in MODEL_KEYS
    }
    embeddings = {key: model.encode(texts, convert_to_numpy=True) for key, model in encoders.items()}
    splitter = StratifiedKFold(n_splits=int(config["folds"]), shuffle=True, random_state=int(config["seed"]))
    rows = []
    for voter, (encoder_key, estimator) in voter_specs(int(config["seed"])).items():
        for fold, (train_index, test_index) in enumerate(splitter.split(embeddings[encoder_key], labels), start=1):
            fitted = deepcopy(estimator).fit(embeddings[encoder_key][train_index], labels[train_index])
            prediction = fitted.predict(embeddings[encoder_key][test_index])
            rows.append({
                "voter": voter,
                "fold": fold,
                "accuracy": accuracy_score(labels[test_index], prediction),
                "macro_f1": f1_score(labels[test_index], prediction, average="macro"),
            })
        final = deepcopy(estimator).fit(embeddings[encoder_key], labels)
        joblib.dump(final, output / f"{voter}.joblib")
    pd.DataFrame(rows).to_csv(output / "ten_fold_metrics.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.training_csv, args.config, args.output)


if __name__ == "__main__":
    main()
