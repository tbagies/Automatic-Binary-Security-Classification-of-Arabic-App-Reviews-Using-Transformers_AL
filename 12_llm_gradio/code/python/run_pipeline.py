from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = Path(os.environ.get("LOCALAPPDATA", PROJECT_ROOT)) / "CAMeLBERT_1NN_Gradio"
RUNTIME_ROOT = Path(os.environ.get("LLM_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT))
os.environ.setdefault("HF_HOME", str(RUNTIME_ROOT / "huggingface"))

import numpy as np
import pandas as pd

from llm_demo.core import (
    CameLBERTEncoder,
    build_heldout_predictions,
    classify_embeddings,
    compute_metrics,
    leave_one_out_neighbors,
    load_datasets,
    select_examples_by_semantic_theme,
)


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAMeLBERT semantic 1-NN pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.json"))
    parser.add_argument("--stage", choices=["predict", "explain", "visuals", "all"], default="all")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--examples-per-group", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def prediction_stage(config: dict, rebuild_cache: bool = False) -> pd.DataFrame:
    cache_dir = PROJECT_ROOT / config["cache_dir"]
    output_dir = PROJECT_ROOT / config["output_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_datasets(config["training_csv"], config["heldout_csv"])
    training = bundle.training
    heldout = bundle.heldout
    training_cache = cache_dir / "training_embeddings.npy"
    heldout_cache = cache_dir / "heldout_embeddings.npy"
    training_table = cache_dir / "training_deduplicated.csv"
    training.to_csv(training_table, index=False, encoding="utf-8-sig")

    encoder = CameLBERTEncoder(config["camelbert_model"], device=config.get("device"))
    batch_size = int(config["embedding_batch_size"])
    if rebuild_cache or not training_cache.exists() or not heldout_cache.exists():
        training_embeddings = encoder.encode(training["content"], batch_size=batch_size)
        heldout_embeddings = encoder.encode(heldout["review"], batch_size=batch_size)
        np.save(training_cache, training_embeddings)
        np.save(heldout_cache, heldout_embeddings)
    else:
        training_embeddings = np.load(training_cache)
        heldout_embeddings = np.load(heldout_cache)

    if training_embeddings.shape != (len(training), 768):
        raise ValueError(f"unexpected training embedding shape: {training_embeddings.shape}")
    if heldout_embeddings.shape != (len(heldout), 768):
        raise ValueError(f"unexpected heldout embedding shape: {heldout_embeddings.shape}")
    norms = np.linalg.norm(training_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4):
        raise ValueError("training embeddings are not L2-normalized")

    predicted, nearest_indexes, similarities = classify_embeddings(
        heldout_embeddings,
        training_embeddings,
        training["Keyword"].to_numpy(),
        batch_size=int(config["similarity_batch_size"]),
    )
    predictions = build_heldout_predictions(heldout, training, predicted, nearest_indexes, similarities)
    predictions.to_csv(output_dir / "heldout_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = compute_metrics(predictions)
    save_json(output_dir / "metrics.json", metrics)

    loo_indexes, loo_similarities = leave_one_out_neighbors(training_embeddings)
    loo_nearest = training.iloc[loo_indexes].reset_index(drop=True)
    training_candidates = pd.DataFrame(
        {
            "source_kind": "training",
            "source_row_index": training["source_row_index"].to_numpy(),
            "review": training["content"].to_numpy(),
            "known_label": training["Keyword"].to_numpy(),
            "predicted_label": loo_nearest["Keyword"].to_numpy(),
            "true_label": training["Keyword"].to_numpy(),
            "cosine_similarity": loo_similarities,
            "nearest_training_row_id": loo_nearest["training_row_id"].to_numpy(),
            "nearest_training_label": loo_nearest["Keyword"].to_numpy(),
            "nearest_training_text": loo_nearest["content"].to_numpy(),
        }
    )
    training_candidates = training_candidates.loc[
        training_candidates["known_label"].eq(training_candidates["predicted_label"])
    ]
    heldout_candidates = pd.DataFrame(
        {
            "source_kind": "heldout",
            "source_row_index": predictions["heldout_row_index"].to_numpy(),
            "review": predictions["review"].to_numpy(),
            "known_label": "",
            "predicted_label": predictions["predicted_label"].to_numpy(),
            "true_label": predictions["true_label"].to_numpy(),
            "cosine_similarity": predictions["cosine_similarity"].to_numpy(),
            "nearest_training_row_id": predictions["nearest_training_row_id"].to_numpy(),
            "nearest_training_label": predictions["nearest_training_label"].to_numpy(),
            "nearest_training_text": predictions["nearest_training_text"].to_numpy(),
        }
    )
    selected = select_examples_by_semantic_theme(training_candidates, heldout_candidates)
    selected.to_csv(output_dir / "selected_examples.csv", index=False, encoding="utf-8-sig")
    save_json(
        output_dir / "run_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_csv": str(Path(config["training_csv"]).resolve()),
            "heldout_csv": str(Path(config["heldout_csv"]).resolve()),
            "camelbert_model": str(Path(config["camelbert_model"]).resolve()),
            "qwen_model": config["qwen_model"],
            "training_rows_original": len(training) + bundle.duplicate_rows_removed,
            "training_rows_deduplicated": len(training),
            "duplicate_rows_removed": bundle.duplicate_rows_removed,
            "heldout_rows": len(heldout),
            "embedding_dimension": encoder.embedding_dimension,
            "classification": "semantic 1-NN cosine similarity",
            "true_label_used_for_inference": False,
            "examples_per_group": int(config["examples_per_group"]),
            "python": sys.version,
            "platform": platform.platform(),
        },
    )
    return selected


def explanation_stage(config: dict) -> pd.DataFrame:
    output_path = PROJECT_ROOT / config["output_dir"] / "selected_examples.csv"
    if not output_path.exists():
        raise FileNotFoundError("Run the predict stage before the explain stage")
    selected = pd.read_csv(output_path)
    if "explanation_ar" in selected.columns and selected["explanation_ar"].fillna("").str.strip().ne("").all():
        return selected
    from llm_demo.qwen import QwenExplainer

    explainer = QwenExplainer(config["qwen_model"], int(config["qwen_max_new_tokens"]))
    selected = explainer.explain_frame(selected)
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")
    return selected


def visual_stage(config: dict) -> Path:
    selected_path = PROJECT_ROOT / config["output_dir"] / "selected_examples.csv"
    selected = pd.read_csv(selected_path)
    if "explanation_ar" not in selected.columns or selected["explanation_ar"].fillna("").str.strip().eq("").any():
        raise ValueError("Qwen explanations are required before visual export")
    from llm_demo.visuals import export_vector_artifacts

    return export_vector_artifacts(selected, PROJECT_ROOT / config["output_dir"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size:
        config["embedding_batch_size"] = args.batch_size
    if args.examples_per_group:
        config["examples_per_group"] = args.examples_per_group
    if args.stage in {"predict", "all"}:
        prediction_stage(config, rebuild_cache=args.rebuild_cache)
    if args.stage in {"explain", "all"}:
        explanation_stage(config)
    if args.stage in {"visuals", "all"}:
        print(f"Created {visual_stage(config)}")


if __name__ == "__main__":
    main()
