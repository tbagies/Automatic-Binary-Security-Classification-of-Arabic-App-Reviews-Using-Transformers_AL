from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

LABELS = ("security", "non-security")


@dataclass(frozen=True)
class DatasetBundle:
    training: pd.DataFrame
    heldout: pd.DataFrame
    duplicate_rows_removed: int


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_datasets(training_path: str | Path, heldout_path: str | Path) -> DatasetBundle:
    training = pd.read_csv(training_path)
    heldout = pd.read_csv(heldout_path)
    _require_columns(training, ["content", "Keyword", "training_row_id"], "training data")
    _require_columns(heldout, ["review", "true_label"], "heldout data")

    for name, frame, text_column in (
        ("training data", training, "content"),
        ("heldout data", heldout, "review"),
    ):
        if frame[text_column].isna().any() or frame[text_column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains missing or blank text")

    invalid_training = sorted(set(training["Keyword"].astype(str)) - set(LABELS))
    invalid_heldout = sorted(set(heldout["true_label"].astype(str)) - set(LABELS))
    if invalid_training:
        raise ValueError(f"training data contains unsupported labels: {invalid_training}")
    if invalid_heldout:
        raise ValueError(f"heldout data contains unsupported labels: {invalid_heldout}")

    conflicts = (
        training.groupby("content", sort=False)["Keyword"]
        .nunique()
        .loc[lambda counts: counts > 1]
    )
    if not conflicts.empty:
        raise ValueError(
            "conflicting labels found for duplicate training texts: "
            + json.dumps(conflicts.index.tolist()[:10], ensure_ascii=False)
        )

    training = training.copy()
    heldout = heldout.copy()
    training.insert(0, "source_row_index", np.arange(len(training), dtype=np.int64))
    heldout.insert(0, "heldout_row_index", np.arange(len(heldout), dtype=np.int64))
    before = len(training)
    training = training.drop_duplicates(subset=["content"], keep="first").reset_index(drop=True)
    heldout = heldout.reset_index(drop=True)
    return DatasetBundle(training, heldout, before - len(training))


class CameLBERTEncoder:
    """Load the saved CAMeLBERT transformer and reproduce mean pooling."""

    def __init__(self, model_path: str | Path, device: str | None = None, max_length: int = 512):
        self.model_path = str(Path(model_path))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @property
    def embedding_dimension(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        text_list = [str(text) for text in texts]
        batches: list[np.ndarray] = []
        for start in range(0, len(text_list), batch_size):
            encoded = self.tokenizer(
                text_list[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.float().cpu().numpy())
        if not batches:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        result = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("encoder produced non-finite embeddings")
        return result


def classify_embeddings(
    query_embeddings: np.ndarray,
    training_embeddings: np.ndarray,
    training_labels: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if query_embeddings.ndim != 2 or training_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if query_embeddings.shape[1] != training_embeddings.shape[1]:
        raise ValueError("query and training embedding dimensions differ")
    nearest: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    for start in range(0, len(query_embeddings), batch_size):
        scores = query_embeddings[start : start + batch_size] @ training_embeddings.T
        indexes = np.argmax(scores, axis=1)
        nearest.append(indexes)
        similarities.append(scores[np.arange(len(indexes)), indexes])
    nearest_indexes = np.concatenate(nearest).astype(np.int64) if nearest else np.array([], dtype=np.int64)
    nearest_similarities = (
        np.concatenate(similarities).astype(np.float32) if similarities else np.array([], dtype=np.float32)
    )
    predicted_labels = training_labels[nearest_indexes]
    return predicted_labels, nearest_indexes, nearest_similarities


def leave_one_out_neighbors(embeddings: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    indexes_out: list[np.ndarray] = []
    similarities_out: list[np.ndarray] = []
    total = len(embeddings)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        scores = embeddings[start:stop] @ embeddings.T
        local_rows = np.arange(stop - start)
        scores[local_rows, np.arange(start, stop)] = -np.inf
        indexes = np.argmax(scores, axis=1)
        indexes_out.append(indexes)
        similarities_out.append(scores[local_rows, indexes])
    return np.concatenate(indexes_out), np.concatenate(similarities_out).astype(np.float32)


def build_heldout_predictions(
    heldout: pd.DataFrame,
    training: pd.DataFrame,
    predicted_labels: np.ndarray,
    nearest_indexes: np.ndarray,
    similarities: np.ndarray,
) -> pd.DataFrame:
    nearest = training.iloc[nearest_indexes].reset_index(drop=True)
    result = heldout.copy()
    result["predicted_label"] = predicted_labels
    result["cosine_similarity"] = similarities
    result["nearest_training_row_id"] = nearest["training_row_id"].to_numpy()
    result["nearest_training_source_row_index"] = nearest["source_row_index"].to_numpy()
    result["nearest_training_label"] = nearest["Keyword"].to_numpy()
    result["nearest_training_text"] = nearest["content"].to_numpy()
    result["is_correct"] = result["predicted_label"].eq(result["true_label"])
    return result


def select_examples(
    training: pd.DataFrame,
    heldout_predictions: pd.DataFrame,
    training_embeddings: np.ndarray,
    examples_per_group: int = 3,
) -> pd.DataFrame:
    loo_indexes, loo_similarities = leave_one_out_neighbors(training_embeddings)
    nearest = training.iloc[loo_indexes].reset_index(drop=True)
    training_candidates = pd.DataFrame(
        {
            "source_kind": "training",
            "source_row_index": training["source_row_index"].to_numpy(),
            "review": training["content"].to_numpy(),
            "known_label": training["Keyword"].to_numpy(),
            "predicted_label": nearest["Keyword"].to_numpy(),
            "true_label": training["Keyword"].to_numpy(),
            "cosine_similarity": loo_similarities,
            "nearest_training_row_id": nearest["training_row_id"].to_numpy(),
            "nearest_training_label": nearest["Keyword"].to_numpy(),
            "nearest_training_text": nearest["content"].to_numpy(),
        }
    )
    training_candidates = training_candidates.loc[
        training_candidates["known_label"].eq(training_candidates["predicted_label"])
    ]

    heldout_candidates = pd.DataFrame(
        {
            "source_kind": "heldout",
            "source_row_index": heldout_predictions["heldout_row_index"].to_numpy(),
            "review": heldout_predictions["review"].to_numpy(),
            "known_label": "",
            "predicted_label": heldout_predictions["predicted_label"].to_numpy(),
            "true_label": heldout_predictions["true_label"].to_numpy(),
            "cosine_similarity": heldout_predictions["cosine_similarity"].to_numpy(),
            "nearest_training_row_id": heldout_predictions["nearest_training_row_id"].to_numpy(),
            "nearest_training_label": heldout_predictions["nearest_training_label"].to_numpy(),
            "nearest_training_text": heldout_predictions["nearest_training_text"].to_numpy(),
        }
    )

    selected: list[pd.DataFrame] = []
    for label in LABELS:
        train_part = (
            training_candidates.loc[training_candidates["known_label"].eq(label)]
            .sort_values(["cosine_similarity", "source_row_index"], ascending=[False, True], kind="mergesort")
            .head(examples_per_group)
        )
        heldout_part = (
            heldout_candidates.loc[heldout_candidates["predicted_label"].eq(label)]
            .sort_values(["cosine_similarity", "source_row_index"], ascending=[False, True], kind="mergesort")
            .head(examples_per_group)
        )
        if len(train_part) != examples_per_group or len(heldout_part) != examples_per_group:
            raise ValueError(f"could not select {examples_per_group} training and heldout examples for {label}")
        selected.extend([train_part, heldout_part])
    output = pd.concat(selected, ignore_index=True)
    output.insert(0, "example_id", "")
    counters: dict[tuple[str, str], int] = {}
    for index, row in output.iterrows():
        key = (str(row["predicted_label"]), str(row["source_kind"]))
        counters[key] = counters.get(key, 0) + 1
        output.at[index, "example_id"] = f"{key[0]}_{key[1]}_{counters[key]:02d}"
    return output


SEMANTIC_THEME_SPECS = {
    ("security", "training"): [
        {
            "semantic_theme": "Privacy",
            "semantic_theme_ar": "الخصوصية",
            "include": r"خصوص|تجسس|بيانات شخصية|بياناتي",
            "exclude": r"احتيال|نصب|رمز|دخول|بنك|بطاق",
            "min_length": 20,
            "max_length": 100,
        },
        {
            "semantic_theme": "Authentication",
            "semantic_theme_ar": "المصادقة وتسجيل الدخول",
            "include": r"رمز التحقق|تسجيل الدخول|كلمة المرور|بصم",
            "exclude": r"احتيال|نصب|تجسس|خصوص",
            "min_length": 10,
            "max_length": 180,
        },
        {
            "semantic_theme": "Fraud",
            "semantic_theme_ar": "الاحتيال",
            "include": r"احتيال|نصب",
            "exclude": r"تجسس|خصوص|رمز|دخول|حساب|بنك|بطاق",
            "min_length": 5,
            "max_length": 180,
        },
    ],
    ("security", "heldout"): [
        {
            "semantic_theme": "Payment verification",
            "semantic_theme_ar": "التحقق من الدفع والبطاقات",
            "include": r"تسجيل البطاقة|بطاق.*رمز|دفع.*رمز|فشل.*بطاق",
            "exclude": "",
            "min_length": 10,
            "max_length": 220,
        },
        {
            "semantic_theme": "Unauthorized charges",
            "semantic_theme_ar": "السحب غير المصرح به",
            "include": r"سحب.*بدون اذن|سحب.*بدون إذن|سحب.*دون علم|يسحبو دون موافقتك|خصم.*بدون",
            "exclude": "",
            "min_length": 10,
            "max_length": 260,
        },
        {
            "semantic_theme": "Data protection",
            "semantic_theme_ar": "حماية البيانات والتشفير",
            "include": r"بدون تشفير|تشفير|حفاظ على الخصوصية|حماية البيانات",
            "exclude": r"بطاق|دفع|خصم|تسجيل الدخول|رمز التحقق|احتيال|نصب",
            "min_length": 10,
            "max_length": 220,
        },
    ],
    ("non-security", "training"): [
        {
            "semantic_theme": "Usability",
            "semantic_theme_ar": "سهولة الاستخدام",
            "include": r"سهل|سهولة",
            "exclude": r"لا يعمل|تحديث|صعب|معقد",
            "min_length": 10,
            "max_length": 160,
        },
        {
            "semantic_theme": "Performance",
            "semantic_theme_ar": "الأداء والسرعة",
            "include": r"بطي|سريع|تعليق|يعلق",
            "exclude": r"تحديث|دعم|اعلان|إعلان",
            "min_length": 10,
            "max_length": 160,
        },
        {
            "semantic_theme": "Updates",
            "semantic_theme_ar": "التحديثات",
            "include": r"تحديث|اصدار|إصدار",
            "exclude": r"دخول|خصوص|حساب|كلمة المرور|رمز التحقق",
            "min_length": 10,
            "max_length": 160,
        },
    ],
    ("non-security", "heldout"): [
        {
            "semantic_theme": "Advertising",
            "semantic_theme_ar": "الإعلانات",
            "include": r"اعلان|إعلان|دعاي",
            "exclude": "",
            "min_length": 10,
            "max_length": 180,
        },
        {
            "semantic_theme": "Language support",
            "semantic_theme_ar": "دعم اللغة العربية",
            "include": r"اللغة العربية|لغة عربية|يدعم.*العربية",
            "exclude": "",
            "min_length": 10,
            "max_length": 180,
        },
        {
            "semantic_theme": "Customer support",
            "semantic_theme_ar": "دعم العملاء",
            "include": r"الدعم|خدمة العملاء",
            "exclude": r"اختراق|خصوص|تجسس|احتيال|نصب",
            "min_length": 10,
            "max_length": 180,
        },
    ],
}


def select_examples_by_semantic_theme(
    training_candidates: pd.DataFrame,
    heldout_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Select one high-support example for each of 12 distinct, explicit themes."""
    selected_rows: list[pd.Series] = []
    used_texts: set[str] = set()
    for label in LABELS:
        for source_kind, candidates, label_column in (
            ("training", training_candidates, "known_label"),
            ("heldout", heldout_candidates, "predicted_label"),
        ):
            base = candidates.loc[candidates[label_column].eq(label)].copy()
            for spec in SEMANTIC_THEME_SPECS[(label, source_kind)]:
                text = base["review"].astype(str)
                mask = text.str.contains(spec["include"], regex=True, case=False, na=False)
                if spec["exclude"]:
                    mask &= ~text.str.contains(spec["exclude"], regex=True, case=False, na=False)
                mask &= text.str.len().between(spec["min_length"], spec["max_length"])
                mask &= ~text.isin(used_texts)
                matches = base.loc[mask].sort_values(
                    ["cosine_similarity", "source_row_index"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                if matches.empty:
                    raise ValueError(
                        f"No eligible {label}/{source_kind} example for theme {spec['semantic_theme']}"
                    )
                row = matches.iloc[0].copy()
                row["semantic_theme"] = spec["semantic_theme"]
                row["semantic_theme_ar"] = spec["semantic_theme_ar"]
                selected_rows.append(row)
                used_texts.add(str(row["review"]))

    output = pd.DataFrame(selected_rows).reset_index(drop=True)
    output.insert(0, "example_id", "")
    counters: dict[tuple[str, str], int] = {}
    for index, row in output.iterrows():
        key = (str(row["predicted_label"]), str(row["source_kind"]))
        counters[key] = counters.get(key, 0) + 1
        output.at[index, "example_id"] = f"{key[0]}_{key[1]}_{counters[key]:02d}"
    if output["semantic_theme"].nunique() != 12 or output["review"].nunique() != 12:
        raise ValueError("Semantic theme selection did not produce 12 unique themes and reviews")
    return output


def compute_metrics(predictions: pd.DataFrame) -> dict:
    labels = list(LABELS)
    confusion = pd.crosstab(predictions["true_label"], predictions["predicted_label"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    per_class = {}
    for label in labels:
        tp = int(confusion.loc[label, label])
        fp = int(confusion[label].sum() - tp)
        fn = int(confusion.loc[label].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion.loc[label].sum()),
        }
    return {
        "accuracy": float(predictions["is_correct"].mean()),
        "total": int(len(predictions)),
        "correct": int(predictions["is_correct"].sum()),
        "confusion_matrix": {
            true_label: {predicted: int(confusion.loc[true_label, predicted]) for predicted in labels}
            for true_label in labels
        },
        "per_class": per_class,
    }
