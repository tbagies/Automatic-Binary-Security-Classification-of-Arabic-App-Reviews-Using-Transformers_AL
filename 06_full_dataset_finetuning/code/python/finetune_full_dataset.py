"""Fine-tune and save the three paper SentenceTransformers on all 10,632 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import BatchAllTripletLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


LABEL_TO_ID = {"non-security": 0, "security": 1}
OUTPUT_NAMES = {
    "camelbert": "CAMeLBERT_Full_Model",
    "marbertv2": "MARBERTv2_Full_Model",
    "paraphrase": "paraphrase-multilingual-MiniLM-L12-v2_Full_Model",
}


def run(training_csv: Path, config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(training_csv)
    required = {"content", "Keyword", "store", "app_name"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if len(frame) != 10_632:
        raise AssertionError("full fine-tuning requires exactly 10,632 rows")
    labels = frame["Keyword"].map(LABEL_TO_ID)
    if labels.value_counts().to_dict() != {0: 5_316, 1: 5_316}:
        raise AssertionError("full fine-tuning data must be balanced 5,316/5,316")

    fine_tuning = config["final_fine_tuning"]
    seed = int(fine_tuning.get("seed", config["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_dict({
        "sentence": frame["content"].astype(str).tolist(),
        "label": labels.astype(int).tolist(),
    })

    for model_key, model_name in config["transformers"].items():
        torch.manual_seed(seed)
        model = SentenceTransformer(model_name)
        model.max_seq_length = min(
            model.max_seq_length, int(fine_tuning.get("max_length", model.max_seq_length))
        )
        model_dir = output / OUTPUT_NAMES[model_key]
        arguments = SentenceTransformerTrainingArguments(
            output_dir=str(model_dir / "training"),
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
            train_dataset=dataset,
            loss=BatchAllTripletLoss(model=model),
        )
        trainer.train()
        trainer.save_model(model_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.training_csv, args.config, args.output)


if __name__ == "__main__":
    main()
