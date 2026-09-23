from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .explanations import build_explanation_messages, finalize_explanation


@dataclass
class QwenExplainer:
    model_name: str = "Qwen/Qwen3-8B"
    max_new_tokens: int = 160

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": "cpu"}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self.model.eval()

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^\s*(?:الشرح|التفسير|Explanation)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        return " ".join(text.strip().split())

    def explain(self, review: str, predicted_label: str) -> str:
        messages = build_explanation_messages(review, predicted_label)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        candidate = self._clean(self.tokenizer.decode(generated, skip_special_tokens=True))
        return finalize_explanation(review, predicted_label, candidate)

    def explain_frame(self, selected: pd.DataFrame) -> pd.DataFrame:
        output = selected.copy()
        explanations = []
        for row in output.itertuples(index=False):
            explanations.append(self.explain(str(row.review), str(row.predicted_label)))
        output["explanation_ar"] = explanations
        return output
