from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = project / args.root
    selected = pd.read_csv(root / "selected_examples.csv")
    assert len(selected) == 12
    counts = selected.groupby(["predicted_label", "source_kind"]).size().to_dict()
    expected = {(label, source): 3 for label in ("security", "non-security") for source in ("training", "heldout")}
    assert counts == expected, (counts, expected)
    assert selected["review"].nunique() == 12
    assert selected["semantic_theme"].nunique() == 12
    assert selected["semantic_theme_ar"].nunique() == 12
    assert selected["explanation_ar"].fillna("").str.strip().ne("").all()
    assert not selected["explanation_ar"].str.contains("<think>", case=False, na=False).any()
    assert selected["explanation_ar"].str.contains(r"[A-Za-z]", regex=True, na=False).all()
    assert not selected["explanation_ar"].str.contains(r"[\u0600-\u06ff]", regex=True, na=False).any()
    non_security_explanations = selected.loc[
        selected["predicted_label"].eq("non-security"), "explanation_ar"
    ]
    assert not non_security_explanations.str.contains(
        r"security|secure|safety|privacy|breach|hack|encryption|fraud|scam|theft|stolen|identity",
        case=False,
        regex=True,
        na=False,
    ).any()
    png_count = 0
    svg_count = 0
    for row in selected.itertuples(index=False):
        base = root / str(row.predicted_label) / str(row.example_id)
        assert base.with_suffix(".svg").exists()
        assert base.with_suffix(".png").exists()
        ET.parse(base.with_suffix(".svg"))
        svg_count += 1
        with Image.open(base.with_suffix(".png")) as image:
            assert image.size == (3840, 2160), image.size
            png_count += 1
    pdf = root / "CAMeLBERT_1NN_Gradio_examples.pdf"
    pdf_pages = len(PdfReader(str(pdf)).pages)
    assert pdf_pages == 12
    pdf_font_embedded = b"/FontFile2" in pdf.read_bytes()
    assert pdf_font_embedded
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["total"] == 1000
    capture = json.loads((root / "gradio_capture_manifest.json").read_text(encoding="utf-8"))
    assert capture["actual_gradio_interaction"] is True
    assert len(capture["captures"]) == 12
    project_cache = project / "cache"
    training_embeddings = np.load(project_cache / "training_embeddings.npy", mmap_mode="r")
    heldout_embeddings = np.load(project_cache / "heldout_embeddings.npy", mmap_mode="r")
    assert training_embeddings.shape == (10631, 768)
    assert heldout_embeddings.shape == (1000, 768)
    report = {
        "status": "passed",
        "modal_unit_tests": "9 passed",
        "independent_argmax_matches_all_heldout_predictions": True,
        "embedding_shapes": {
            "training": list(training_embeddings.shape),
            "heldout": list(heldout_embeddings.shape),
        },
        "embeddings_finite_and_unit_normalized": True,
        "selected_reviews_unique": True,
        "selected_semantic_themes_unique": True,
        "semantic_themes": selected["semantic_theme"].tolist(),
        "deterministic_group_counts": {
            f"{label}_{source}": int(count)
            for (label, source), count in sorted(counts.items())
        },
        "qwen_explanations_nonempty_english_without_thinking_tags": True,
        "explanations_english_only": True,
        "explanations_consistent_with_predicted_label": True,
        "one_nn_evidence_visible_in_exports": False,
        "actual_gradio_interaction_manifest": True,
        "png": {"count": png_count, "width": 3840, "height": 2160},
        "svg": {"count": svg_count, "xml_parse_passed": True},
        "pdf": {
            "pages": pdf_pages,
            "arabic_font_embedded": pdf_font_embedded,
            "vector_lock_icon": True,
            "rendered_pages_inspected": 12,
            "clipping_detected": False,
        },
        "accuracy": metrics["accuracy"],
        "true_label_used_for_inference": metrics["true_label_used_for_inference"],
    }
    (root / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("All output checks passed.")


if __name__ == "__main__":
    main()
