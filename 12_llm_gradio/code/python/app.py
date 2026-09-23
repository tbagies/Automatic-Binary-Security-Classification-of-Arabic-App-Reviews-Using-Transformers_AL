from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = Path(os.environ.get("LOCALAPPDATA", PROJECT_ROOT)) / "CAMeLBERT_1NN_Gradio"
RUNTIME_ROOT = Path(os.environ.get("LLM_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT))
os.environ.setdefault("HF_HOME", str(RUNTIME_ROOT / "huggingface"))

import gradio as gr
import numpy as np
import pandas as pd

from llm_demo.core import CameLBERTEncoder, classify_embeddings


class DemoService:
    def __init__(self, config_path: str | Path):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cache_dir = PROJECT_ROOT / self.config["cache_dir"]
        output_dir = PROJECT_ROOT / self.config["output_dir"]
        self.training = pd.read_csv(cache_dir / "training_deduplicated.csv")
        self.training_embeddings = np.load(cache_dir / "training_embeddings.npy")
        self.selected = pd.read_csv(output_dir / "selected_examples.csv")
        self.encoder = CameLBERTEncoder(self.config["camelbert_model"], device=self.config.get("device"))
        self.explanation_cache = {
            str(row.review): str(row.explanation_ar)
            for row in self.selected.itertuples(index=False)
            if str(getattr(row, "explanation_ar", "")).strip()
        }
        self._qwen = None

    def _get_explainer(self):
        if self._qwen is None:
            from llm_demo.qwen import QwenExplainer

            self._qwen = QwenExplainer(
                self.config["qwen_model"],
                int(self.config["qwen_max_new_tokens"]),
            )
        return self._qwen

    def analyze(self, text: str):
        text = str(text or "").strip()
        if not text:
            return "Please enter a review", "", "⚠️ Enter an Arabic review to begin."
        try:
            query = self.encoder.encode([text], batch_size=1)
            predicted, _, _ = classify_embeddings(
                query,
                self.training_embeddings,
                self.training["Keyword"].to_numpy(),
            )
            label = str(predicted[0])
            explanation = self.explanation_cache.get(text)
            if not explanation:
                explanation = self._get_explainer().explain(text, label)
            return label, explanation, "Analysis complete / اكتمل التحليل"
        except Exception as exc:
            return "Error", "", f"❌ Analysis failed: {html.escape(str(exc))}"


CSS = """
:root { color-scheme: dark; }
html, body, .gradio-container { background: #0d0d0f !important; color: #f7f7f8 !important; }
body { overflow-x: hidden !important; }
.gradio-container { max-width: 1920px !important; margin: auto !important; padding: 6px 16px 28px !important; }
#capture-panel { background: #0d0d0f !important; border: 0 !important; padding: 0 !important; min-height: 1080px; }
#capture-panel > div, #app-title { background: #0d0d0f !important; }
#app-title h1 { font-family: Arial, "Noto Color Emoji", sans-serif !important; font-size: 68px !important; line-height: 1.15 !important; white-space: normal !important; margin: 0 0 64px !important; color: #f7f7f8 !important; }
#app-title p { font-size: 34px !important; line-height: 1.45 !important; margin: 0 0 50px !important; color: #f7f7f8 !important; }
#app-title strong { color: #f8fafc !important; }
#input-card, #output-card { background: #27272a !important; border: 3px solid #414148 !important; border-radius: 14px !important; padding: 30px 34px 28px !important; }
#review-input, #review-input > div, #predicted-label, #predicted-label > div, #qwen-explanation, #qwen-explanation > div { background: #27272a !important; color: #f8fafc !important; }
#review-input label, #predicted-label label, #qwen-explanation label { color: #f3f4f6 !important; font-size: 34px !important; margin-bottom: 20px !important; }
#review-input textarea, #predicted-label input, #qwen-explanation textarea { background: #27272a !important; color: #f8fafc !important; border: 3px solid #414148 !important; border-radius: 12px !important; }
#review-input textarea { direction: rtl !important; text-align: right !important; font-size: 38px !important; line-height: 1.55 !important; min-height: 310px !important; padding: 28px !important; }
#predicted-label input { font-size: 36px !important; font-weight: 500 !important; min-height: 104px !important; padding: 26px !important; }
#qwen-explanation textarea { direction: ltr !important; text-align: left !important; font-size: 34px !important; line-height: 1.55 !important; min-height: 275px !important; padding: 26px !important; }
#analyze-button { min-height: 116px; margin-top: 24px !important; font-size: 40px !important; font-weight: 700; background: #f2550b !important; border: 0 !important; }
#analysis-status, #analysis-status * { font-size: 20px !important; color: #cbd5e1 !important; }
#examples-title { margin-top: 28px !important; }
"""


def build_demo(service: DemoService) -> gr.Blocks:
    examples = [[str(text)] for text in service.selected["review"].tolist()]
    with gr.Blocks(title="Security related user reviews Analyzer", css=CSS, theme=gr.themes.Base()) as demo:
        with gr.Group(elem_id="capture-panel"):
            gr.Markdown(
                """
                # 🔒 Security related user reviews Analyzer
                Classify text into security-related categories and get expert-level explanations. **Supports Arabic text.**
                """,
                elem_id="app-title",
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_id="input-card"):
                    review = gr.Textbox(
                        label="Enter your text",
                        placeholder="Enter your user review in Arabic...",
                        lines=5,
                        elem_id="review-input",
                    )
                    analyze = gr.Button("🔍 Analyze", variant="primary", elem_id="analyze-button")
                with gr.Column(scale=1, elem_id="output-card"):
                    predicted = gr.Textbox(label="📊 Predicted Label", interactive=False, elem_id="predicted-label")
                    explanation = gr.Textbox(
                        label="💡 Explanation",
                        lines=5,
                        interactive=False,
                        elem_id="qwen-explanation",
                    )
                    status = gr.Markdown("Ready", elem_id="analysis-status", visible=False)
        gr.Markdown("### 📝 Example Requirements", elem_id="examples-title")
        gr.Examples(examples=examples, inputs=review, label="12 theme-diverse reproducible examples")

        analyze.click(
            fn=service.analyze,
            inputs=review,
            outputs=[predicted, explanation, status],
            api_name="analyze",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CAMeLBERT 1-NN Gradio application")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_demo(DemoService(args.config))
    app.launch(server_name=args.host, server_port=args.port, share=False, inbrowser=False, show_error=True)
