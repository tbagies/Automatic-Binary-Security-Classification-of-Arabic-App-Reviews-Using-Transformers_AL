"""Reproducible 2x browser capture helper for local use.

Codex uses its connected browser for the delivered captures. This script lets a
researcher reproduce the same interaction locally with Playwright.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--selected", default="outputs/selected_examples.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    selected = pd.read_csv(root / args.selected)
    output_root = root / args.output
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle")
        for row in selected.itertuples(index=False):
            page.locator("#review-input textarea").fill(str(row.review))
            page.locator("#analyze-button").click()
            page.locator("#analysis-status").get_by_text("Analysis complete").wait_for(timeout=180_000)
            destination = output_root / str(row.predicted_label) / f"{row.example_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            page.locator("#capture-panel").screenshot(path=str(destination))
        browser.close()


if __name__ == "__main__":
    main()
