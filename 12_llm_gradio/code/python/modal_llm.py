from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "arabic-mobile-app-reviews-llm-1nn-gradio"
DATA_VOLUME_NAME = "arabic-mobile-app-reviews-final-used-data"
HF_CACHE_VOLUME_NAME = "arabic-mobile-app-reviews-final-used-hf-cache"
REMOTE_INPUT_ROOT = PurePosixPath("/data/inputs/app_distribution_validation")
REMOTE_RUN_ROOT = PurePosixPath("/data/outputs/LLM_1NN_Gradio/current")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "chromium",
        "fonts-noto-core",
        "fonts-noto-extra",
        "fonts-noto-color-emoji",
        "poppler-utils",
    )
    .uv_pip_install(
        "torch==2.13.0",
        "sentence-transformers==6.0.0",
        "transformers==5.16.1",
        "accelerate==1.14.0",
        "safetensors==0.8.0",
        "huggingface-hub==1.29.0",
        "pandas==2.3.3",
        "numpy==2.5.2",
        "gradio==6.26.0",
        "fastapi==0.141.1",
        "reportlab==5.0.1",
        "pypdf==6.16.2",
        "arabic-reshaper==3.0.1",
        "python-bidi==0.6.11",
        "pillow==12.3.0",
        "playwright==1.62.0",
        "pytest==9.0.2",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    .add_local_dir(
        LOCAL_PROJECT_ROOT,
        "/root/llm_project",
        ignore=["**/__pycache__/**", "**/*.pyc", "cache/**", "outputs/**"],
    )
)

app = modal.App(APP_NAME)
COMMON = {
    "image": image,
    "gpu": ["L40S", "A10", "L4"],
    "cpu": 16,
    "memory": 65_536,
    "timeout": 24 * 60 * 60,
    "volumes": {
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
}


def _required_remote_paths() -> list[PurePosixPath]:
    return [
        REMOTE_INPUT_ROOT / "final_label_data.csv",
        REMOTE_INPUT_ROOT / "heldout.csv",
        REMOTE_INPUT_ROOT / "Saved_Model" / "CAMeLBERT_Full_Model" / "model.safetensors",
        REMOTE_INPUT_ROOT / "Saved_Model" / "CAMeLBERT_Full_Model" / "modules.json",
    ]


def _clean_qwen(text: str) -> str:
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*(?:الشرح|التفسير|Explanation)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.strip().split())


def _generate_explanation(tokenizer, model, review: str, label: str) -> str:
    import torch
    from llm_demo.explanations import build_explanation_messages, finalize_explanation

    messages = build_explanation_messages(review, label)
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    candidate = _clean_qwen(
        tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    )
    return finalize_explanation(review, label, candidate)


def _compute_pipeline() -> tuple[object, object, object, dict, object, object]:
    import sys

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    input_root = Path(str(REMOTE_INPUT_ROOT))
    required_paths = [Path(str(path)) for path in _required_remote_paths()]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Modal volume inputs:\n" + "\n".join(missing))
    train_path = input_root / "final_label_data.csv"
    heldout_path = input_root / "heldout.csv"
    model_path = input_root / "Saved_Model" / "CAMeLBERT_Full_Model"
    training = pd.read_csv(train_path)
    heldout = pd.read_csv(heldout_path)
    required_training = {"content", "Keyword", "training_row_id"}
    required_heldout = {"review", "true_label"}
    if not required_training.issubset(training.columns):
        raise ValueError(f"Training columns missing: {sorted(required_training - set(training.columns))}")
    if not required_heldout.issubset(heldout.columns):
        raise ValueError(f"Heldout columns missing: {sorted(required_heldout - set(heldout.columns))}")
    if training["content"].isna().any() or heldout["review"].isna().any():
        raise ValueError("Missing review text found")
    conflicts = training.groupby("content", sort=False)["Keyword"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting labels found for duplicate training text")
    training.insert(0, "source_row_index", np.arange(len(training), dtype=np.int64))
    heldout.insert(0, "heldout_row_index", np.arange(len(heldout), dtype=np.int64))
    original_training_rows = len(training)
    training = training.drop_duplicates("content", keep="first").reset_index(drop=True)
    heldout = heldout.reset_index(drop=True)

    encoder = SentenceTransformer(str(model_path), device="cuda")
    train_embeddings = encoder.encode(
        training["content"].astype(str).tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    heldout_embeddings = encoder.encode(
        heldout["review"].astype(str).tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    if train_embeddings.shape != (len(training), 768):
        raise ValueError(f"Unexpected training embedding shape: {train_embeddings.shape}")
    if heldout_embeddings.shape != (len(heldout), 768):
        raise ValueError(f"Unexpected heldout embedding shape: {heldout_embeddings.shape}")

    train_tensor = torch.from_numpy(train_embeddings).to("cuda")
    heldout_tensor = torch.from_numpy(heldout_embeddings).to("cuda")
    nearest_parts, similarity_parts = [], []
    for start in range(0, len(heldout_tensor), 256):
        scores = heldout_tensor[start : start + 256] @ train_tensor.T
        similarities, indexes = torch.max(scores, dim=1)
        nearest_parts.append(indexes.cpu().numpy())
        similarity_parts.append(similarities.cpu().numpy())
    nearest_indexes = np.concatenate(nearest_parts).astype(np.int64)
    similarities = np.concatenate(similarity_parts).astype(np.float32)
    nearest = training.iloc[nearest_indexes].reset_index(drop=True)
    predictions = heldout.copy()
    predictions["predicted_label"] = nearest["Keyword"].to_numpy()
    predictions["cosine_similarity"] = similarities
    predictions["nearest_training_row_id"] = nearest["training_row_id"].to_numpy()
    predictions["nearest_training_source_row_index"] = nearest["source_row_index"].to_numpy()
    predictions["nearest_training_label"] = nearest["Keyword"].to_numpy()
    predictions["nearest_training_text"] = nearest["content"].to_numpy()
    predictions["is_correct"] = predictions["predicted_label"].eq(predictions["true_label"])

    loo_indexes_parts, loo_similarity_parts = [], []
    for start in range(0, len(train_tensor), 256):
        stop = min(start + 256, len(train_tensor))
        scores = train_tensor[start:stop] @ train_tensor.T
        local_rows = torch.arange(stop - start, device="cuda")
        scores[local_rows, torch.arange(start, stop, device="cuda")] = -torch.inf
        loo_similarities, loo_indexes = torch.max(scores, dim=1)
        loo_indexes_parts.append(loo_indexes.cpu().numpy())
        loo_similarity_parts.append(loo_similarities.cpu().numpy())
    loo_indexes = np.concatenate(loo_indexes_parts).astype(np.int64)
    loo_similarities = np.concatenate(loo_similarity_parts).astype(np.float32)
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
    sys.path.insert(0, "/root/llm_project")
    from llm_demo.core import select_examples_by_semantic_theme

    selected = select_examples_by_semantic_theme(training_candidates, heldout_candidates)
    if len(selected) != 12:
        raise ValueError(f"Expected 12 examples, got {len(selected)}")

    confusion = pd.crosstab(predictions["true_label"], predictions["predicted_label"]).reindex(
        index=["security", "non-security"], columns=["security", "non-security"], fill_value=0
    )
    per_class = {}
    for label in ("security", "non-security"):
        tp = int(confusion.loc[label, label])
        fp = int(confusion[label].sum() - tp)
        fn = int(confusion.loc[label].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion.loc[label].sum())}
    metrics = {
        "accuracy": float(predictions["is_correct"].mean()),
        "total": int(len(predictions)),
        "correct": int(predictions["is_correct"].sum()),
        "confusion_matrix": {
            true_label: {predicted: int(confusion.loc[true_label, predicted]) for predicted in ("security", "non-security")}
            for true_label in ("security", "non-security")
        },
        "per_class": per_class,
        "true_label_used_for_inference": False,
        "training_rows_original": original_training_rows,
        "training_rows_deduplicated": len(training),
    }
    del encoder, train_tensor, heldout_tensor
    torch.cuda.empty_cache()
    return training, predictions, selected, metrics, train_embeddings, heldout_embeddings


class _ModalDemoService:
    def __init__(self, training, train_embeddings, selected, encoder, tokenizer, qwen):
        import numpy as np

        self.training = training
        self.train_embeddings = train_embeddings
        self.selected = selected
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.qwen = qwen
        self.explanations = {
            str(row.review): str(row.explanation_ar) for row in selected.itertuples(index=False)
        }

    def analyze(self, text: str):
        import numpy as np

        text = str(text or "").strip()
        if not text:
            return "Please enter a review", "", "⚠️ Enter an Arabic review to begin."
        query = self.encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        scores = query @ self.train_embeddings.T
        nearest_index = int(np.argmax(scores[0]))
        nearest = self.training.iloc[nearest_index]
        label = str(nearest["Keyword"])
        explanation = self.explanations.get(text) or _generate_explanation(self.tokenizer, self.qwen, text, label)
        return label, explanation, "Analysis complete / اكتمل التحليل"


def _load_models_for_ui(training, train_embeddings, selected):
    import sys
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/llm_project")
    model_path = Path(str(REMOTE_INPUT_ROOT)) / "Saved_Model" / "CAMeLBERT_Full_Model"
    encoder = SentenceTransformer(str(model_path), device="cuda")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    qwen.eval()
    return _ModalDemoService(training, train_embeddings, selected, encoder, tokenizer, qwen)


def _capture_gradio(service, selected, output_dir: Path) -> dict:
    import time
    import sys
    from playwright.sync_api import sync_playwright

    sys.path.insert(0, "/root/llm_project")
    from app import build_demo

    demo = build_demo(service)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True,
        prevent_thread_lock=True,
    )
    time.sleep(4)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path="/usr/bin/chromium",
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        page = context.new_page()
        page.goto("http://127.0.0.1:7860", wait_until="networkidle", timeout=120_000)
        captured = []
        for row in selected.itertuples(index=False):
            page.locator("#review-input textarea").fill(str(row.review))
            page.locator("#analyze-button").click()
            page.locator("#qwen-explanation textarea").wait_for(state="visible", timeout=120_000)
            page.wait_for_function(
                "expected => document.querySelector('#qwen-explanation textarea')?.value === expected",
                arg=str(row.explanation_ar),
                timeout=180_000,
            )
            destination = output_dir / str(row.predicted_label) / f"{row.example_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(destination), full_page=False)
            captured.append(
                {
                    "example_id": str(row.example_id),
                    "file": str(destination.relative_to(output_dir)),
                    "predicted_label": str(row.predicted_label),
                    "semantic_theme": str(row.semantic_theme),
                    "semantic_theme_ar": str(row.semantic_theme_ar),
                }
            )
        browser.close()
    demo.close()
    return {
        "actual_gradio_interaction": True,
        "browser": "system Chromium via Playwright",
        "viewport_css_pixels": {"width": 1920, "height": 1080},
        "device_scale_factor": 2,
        "capture_target": "3840x2160 viewport after the Qwen explanation matched the completed Gradio result",
        "captures": captured,
    }


@app.function(
    image=image,
    cpu=2,
    memory=4_096,
    timeout=30 * 60,
    volumes={"/data": data_volume},
)
def validate_remote_inputs() -> dict:
    import hashlib

    result = {}
    for remote_path in _required_remote_paths():
        path = Path(str(remote_path))
        if not path.exists():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[str(path)] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


@app.function(image=image, cpu=4, memory=8_192, timeout=30 * 60)
def run_unit_tests() -> dict:
    import subprocess

    completed = subprocess.run(
        ["python", "-m", "pytest", "-q", "tests"],
        cwd="/root/llm_project",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout + "\n" + completed.stderr)
    return {"status": "passed", "output": completed.stdout.strip()}


@app.function(**COMMON)
def run_pipeline_and_capture() -> dict:
    import hashlib
    import json
    import sys
    from datetime import datetime, timezone

    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_root = Path(str(REMOTE_RUN_ROOT))
    if run_root.exists():
        shutil.rmtree(run_root)
    cache_dir = run_root / "cache"
    output_dir = run_root / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    training, predictions, selected, metrics, train_embeddings, heldout_embeddings = _compute_pipeline()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    qwen.eval()
    selected["explanation_ar"] = [
        _generate_explanation(tokenizer, qwen, str(row.review), str(row.predicted_label))
        for row in selected.itertuples(index=False)
    ]
    training.to_csv(cache_dir / "training_deduplicated.csv", index=False, encoding="utf-8-sig")
    np.save(cache_dir / "training_embeddings.npy", train_embeddings)
    np.save(cache_dir / "heldout_embeddings.npy", heldout_embeddings)
    predictions.to_csv(output_dir / "heldout_predictions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "selected_examples.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "Modal GPU",
        "gpu": torch.cuda.get_device_name(0),
        "training_csv": str(REMOTE_INPUT_ROOT / "final_label_data.csv"),
        "heldout_csv": str(REMOTE_INPUT_ROOT / "heldout.csv"),
        "camelbert_model": str(REMOTE_INPUT_ROOT / "Saved_Model" / "CAMeLBERT_Full_Model"),
        "qwen_model": "Qwen/Qwen3-8B",
        "classification": "semantic 1-NN cosine similarity",
        "true_label_used_for_inference": False,
        "embedding_dimension": 768,
        "examples_per_group": 3,
        "selection": "one high-support example per explicit semantic theme",
        "semantic_themes": selected[
            ["example_id", "predicted_label", "source_kind", "semantic_theme", "semantic_theme_ar"]
        ].to_dict("records"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    sys.path.insert(0, "/root/llm_project")
    from llm_demo.visuals import export_vector_artifacts

    export_vector_artifacts(selected, output_dir)
    service = _ModalDemoService(training, train_embeddings, selected, None, tokenizer, qwen)
    from sentence_transformers import SentenceTransformer
    service.encoder = SentenceTransformer(
        str(Path(str(REMOTE_INPUT_ROOT)) / "Saved_Model" / "CAMeLBERT_Full_Model"), device="cuda"
    )
    capture_manifest = _capture_gradio(service, selected, output_dir)
    (output_dir / "gradio_capture_manifest.json").write_text(
        json.dumps(capture_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    data_volume.commit()
    hf_cache_volume.commit()
    return {
        "status": "complete",
        "gpu": manifest["gpu"],
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "examples": len(selected),
        "remote_root": str(REMOTE_RUN_ROOT),
    }


@app.function(**COMMON)
@modal.asgi_app()
def gradio_web():
    import sys
    import numpy as np
    import pandas as pd
    from fastapi import FastAPI
    import gradio as gr

    sys.path.insert(0, "/root/llm_project")
    from app import build_demo

    run_root = Path(str(REMOTE_RUN_ROOT))
    training = pd.read_csv(run_root / "cache" / "training_deduplicated.csv")
    train_embeddings = np.load(run_root / "cache" / "training_embeddings.npy")
    selected = pd.read_csv(run_root / "outputs" / "selected_examples.csv")
    service = _load_models_for_ui(training, train_embeddings, selected)
    demo = build_demo(service)
    return gr.mount_gradio_app(FastAPI(), demo, path="/")


def _download_outputs() -> None:
    if not REMOTE_RUN_ROOT.as_posix().startswith("/data/outputs/LLM_1NN_Gradio/"):
        raise RuntimeError("Unsafe remote download root")
    remote_relative_root = REMOTE_RUN_ROOT.relative_to("/data")
    entries = data_volume.listdir("/" + remote_relative_root.as_posix(), recursive=True)
    files = [entry for entry in entries if entry.type.name == "FILE"]
    if not files:
        raise FileNotFoundError(f"No outputs found at {REMOTE_RUN_ROOT}")
    for entry in files:
        remote = PurePosixPath(entry.path)
        prefix = PurePosixPath(remote_relative_root.as_posix())
        relative = remote.relative_to(prefix)
        destination = LOCAL_PROJECT_ROOT.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            for chunk in data_volume.read_file(entry.path):
                stream.write(chunk)
    print(f"Downloaded {len(files)} files into {LOCAL_PROJECT_ROOT}")


def _sha256(path: Path) -> dict:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _validate_local_against_remote() -> dict:
    local_model_candidates = [
        REPOSITORY_ROOT / "06_full_dataset_finetuning" / "models" / "CAMeLBERT_Full_Model",
    ]
    local_model_root = next(
        (
            candidate
            for candidate in local_model_candidates
            if (candidate / "model.safetensors").is_file() and (candidate / "modules.json").is_file()
        ),
        None,
    )
    if local_model_root is None:
        raise FileNotFoundError(
            "CAMeLBERT_Full_Model was not found at the supplied path or its byte-identical current fallback"
        )
    local_paths = {
        str(REMOTE_INPUT_ROOT / "final_label_data.csv"): Path(
            REPOSITORY_ROOT / "04_active_learning" / "iteration_10" / "input" / "training_set.csv"
        ),
        str(REMOTE_INPUT_ROOT / "heldout.csv"): Path(
            REPOSITORY_ROOT / "08_heldout_evaluation" / "data" / "heldout_1000.csv"
        ),
        str(REMOTE_INPUT_ROOT / "Saved_Model" / "CAMeLBERT_Full_Model" / "model.safetensors"): local_model_root / "model.safetensors",
        str(REMOTE_INPUT_ROOT / "Saved_Model" / "CAMeLBERT_Full_Model" / "modules.json"): local_model_root / "modules.json",
    }
    local = {remote: _sha256(local_path) for remote, local_path in local_paths.items()}
    remote = validate_remote_inputs.remote()
    mismatches = {key: {"local": local[key], "remote": remote.get(key)} for key in local if local[key] != remote.get(key)}
    if mismatches:
        raise ValueError("Modal inputs do not match the requested local files: " + json.dumps(mismatches, indent=2))
    return {"status": "matched", "files": local}


@app.local_entrypoint()
def main(job: str = "status", confirm_paid: bool = False) -> None:
    job = job.strip().lower()
    if job == "status":
        print(json.dumps({"inputs": [str(path) for path in _required_remote_paths()], "remote_output": str(REMOTE_RUN_ROOT)}, indent=2))
        return
    if job == "download":
        _download_outputs()
        return
    if job == "preflight":
        print(json.dumps(_validate_local_against_remote(), indent=2))
        return
    if job == "test":
        print(json.dumps(run_unit_tests.remote(), indent=2))
        return
    if job not in {"run", "all"}:
        raise ValueError("job must be status, preflight, test, run, download, or all")
    if not confirm_paid:
        raise RuntimeError("Pass --confirm-paid to authorize Modal GPU compute")
    result = run_pipeline_and_capture.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if job == "all":
        _download_outputs()
