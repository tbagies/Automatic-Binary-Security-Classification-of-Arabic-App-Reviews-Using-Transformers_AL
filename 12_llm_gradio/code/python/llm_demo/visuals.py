from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path

import pandas as pd
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from pypdf import PdfReader
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PAGE_SIZE = landscape((7.5 * inch, 13.333 * inch))


def _wrap_words(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        " ".join(str(text).split()),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def _rtl(text: str) -> str:
    return get_display(reshape(str(text)))


def _output_path(root: Path, row: pd.Series, suffix: str) -> Path:
    label_dir = root / str(row["predicted_label"])
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"{row['example_id']}.{suffix}"


def write_svg(row: pd.Series, destination: Path) -> None:
    review_lines = _wrap_words(str(row["review"]), 40)[:6]
    explanation_lines = _wrap_words(str(row["explanation_ar"]), 45)[:5]
    def svg_text(
        lines: list[str],
        x: int,
        y: int,
        step: int,
        anchor: str = "end",
        direction: str = "rtl",
        css_class: str = "arabic",
    ) -> str:
        return "\n".join(
            f'<text x="{x}" y="{y + index * step}" text-anchor="{anchor}" direction="{direction}" '
            f'unicode-bidi="plaintext" class="{css_class}">{html.escape(line)}</text>'
            for index, line in enumerate(lines)
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
<style>
  .title {{ font: 700 68px Arial, "Segoe UI", sans-serif; fill: #f7f7f8; }}
  .sub {{ font: 34px Arial, "Segoe UI", sans-serif; fill: #f7f7f8; }}
  .label {{ font: 38px Arial, "Segoe UI", sans-serif; fill: #f3f4f6; }}
  .button {{ font: 700 48px Arial, "Segoe UI", sans-serif; fill: #ffffff; }}
  .value {{ font: 38px Arial, "Segoe UI", sans-serif; fill: #f8fafc; }}
  .arabic {{ font: 36px Arial, "Segoe UI", sans-serif; fill: #f8fafc; }}
  .english {{ font: 34px Arial, "Segoe UI", sans-serif; fill: #f8fafc; }}
  .emoji {{ font: 38px "Noto Color Emoji", "Segoe UI Emoji", sans-serif; }}
</style>
<rect width="1920" height="1080" fill="#0d0d0f"/>
<path d="M48 57v-15c0-28 19-43 42-43s42 15 42 43v15" fill="none" stroke="#d8d0e8" stroke-width="13" stroke-linecap="round"/>
<rect x="34" y="52" width="112" height="82" rx="12" fill="#ffb14a"/>
<circle cx="90" cy="91" r="12" fill="#55456e"/><rect x="84" y="98" width="12" height="22" rx="5" fill="#55456e"/>
<text x="165" y="92" class="title">Security related user reviews Analyzer</text>
<text x="16" y="190" class="sub">Classify text into security-related categories and get expert-level explanations. <tspan font-weight="700">Supports Arabic text.</tspan></text>
<rect x="8" y="255" width="910" height="640" rx="14" fill="#27272a" stroke="#414148" stroke-width="3"/>
<text x="44" y="330" class="label">Enter your text</text>
<rect x="44" y="374" width="838" height="330" rx="12" fill="#27272a" stroke="#414148" stroke-width="3"/>
{svg_text(review_lines, 842, 438, 49)}
<rect x="8" y="744" width="910" height="140" rx="14" fill="#f2550b"/>
<circle cx="385" cy="813" r="25" fill="#4db7e8" stroke="#7d4da8" stroke-width="5"/><path d="M403 832l31 31" stroke="#7d4da8" stroke-width="14" stroke-linecap="round"/>
<text x="452" y="832" class="button">Analyze</text>
<rect x="960" y="255" width="952" height="690" rx="14" fill="#27272a" stroke="#414148" stroke-width="3"/>
<text x="1004" y="330" class="emoji">📊</text><text x="1060" y="330" class="label">Predicted Label</text>
<rect x="1000" y="374" width="872" height="110" rx="12" fill="#27272a" stroke="#414148" stroke-width="3"/>
<text x="1032" y="447" class="value">{html.escape(str(row['predicted_label']))}</text>
<text x="1004" y="590" class="emoji">💡</text><text x="1060" y="590" class="label">Explanation</text>
<rect x="1000" y="626" width="872" height="280" rx="12" fill="#27272a" stroke="#414148" stroke-width="3"/>
{svg_text(explanation_lines, 1044, 687, 48, anchor="start", direction="ltr", css_class="english")}
</svg>'''
    destination.write_text(svg, encoding="utf-8")


def _find_arabic_font() -> Path:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Arabic-capable TrueType font was found")


def write_combined_pdf(selected: pd.DataFrame, destination: Path) -> None:
    font_path = _find_arabic_font()
    pdfmetrics.registerFont(TTFont("ArabicUI", str(font_path)))
    page_width, page_height = PAGE_SIZE
    pdf = canvas.Canvas(str(destination), pagesize=PAGE_SIZE, pageCompression=1)
    for _, row in selected.iterrows():
        label_color = HexColor("#ef4444") if row["predicted_label"] == "security" else HexColor("#22c55e")
        pdf.setFillColor(HexColor("#0b0f17")); pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        pdf.setFillColor(HexColor("#f97316")); pdf.roundRect(24, page_height - 72, 48, 48, 10, fill=1, stroke=0)
        # Draw the reference lock as vector geometry. Emoji glyphs are not
        # dependable in PDF renderers and previously degraded to a blank tile.
        pdf.setStrokeColor(white); pdf.setLineWidth(4)
        pdf.arc(35, page_height - 51, 61, page_height - 29, startAng=0, extent=180)
        pdf.setFillColor(white); pdf.setStrokeColor(white)
        pdf.roundRect(33, page_height - 63, 30, 22, 4, fill=1, stroke=0)
        pdf.setFillColor(HexColor("#f97316")); pdf.circle(48, page_height - 51, 2.4, fill=1, stroke=0)
        pdf.rect(46.7, page_height - 57, 2.6, 6, fill=1, stroke=0)
        pdf.setFillColor(white); pdf.setFont("Helvetica-Bold", 28)
        pdf.drawString(86, page_height - 57, "Security related user reviews Analyzer")
        pdf.setFillColor(HexColor("#cbd5e1")); pdf.setFont("Helvetica", 13)
        pdf.drawString(
            26,
            page_height - 94,
            "Classify text into security-related categories and get expert-level explanations. Supports Arabic text.",
        )
        source = "TRAINING EXAMPLE" if row["source_kind"] == "training" else "UNSEEN HELDOUT PREDICTION"
        pdf.setFillColor(HexColor("#172033")); pdf.roundRect(26, page_height - 133, page_width - 52, 24, 7, fill=1, stroke=0)
        pdf.setFillColor(HexColor("#94a3b8")); pdf.setFont("Helvetica", 9)
        pdf.drawString(
            40,
            page_height - 125,
            f"{source}  |  {row['example_id']}  |  Theme: {row.get('semantic_theme', '')}",
        )
        panel_y, panel_h = 36, page_height - 190
        left_x, left_w = 26, page_width * 0.47
        right_x, right_w = page_width * 0.515, page_width * 0.458
        for x, width in ((left_x, left_w), (right_x, right_w)):
            pdf.setFillColor(HexColor("#202632")); pdf.setStrokeColor(HexColor("#3b4557"))
            pdf.roundRect(x, panel_y, width, panel_h, 12, fill=1, stroke=1)
        pdf.setFillColor(white); pdf.setFont("Helvetica-Bold", 18); pdf.drawString(left_x + 20, panel_y + panel_h - 38, "Enter your text")
        pdf.setFillColor(HexColor("#151a24")); pdf.setStrokeColor(HexColor("#465064"))
        pdf.roundRect(left_x + 20, panel_y + 88, left_w - 40, panel_h - 150, 10, fill=1, stroke=1)
        pdf.setFillColor(white); pdf.setFont("ArabicUI", 11.5)
        review_lines = _wrap_words(str(row["review"]), 44)[:9]
        y = panel_y + panel_h - 84
        for line in review_lines:
            pdf.drawRightString(left_x + left_w - 34, y, _rtl(line)); y -= 17
        pdf.setFillColor(HexColor("#f2550b")); pdf.roundRect(left_x + 20, panel_y + 26, left_w - 40, 48, 10, fill=1, stroke=0)
        pdf.setFillColor(white); pdf.setFont("Helvetica-Bold", 18); pdf.drawCentredString(left_x + left_w / 2, panel_y + 42, "Analyze")
        pdf.setFillColor(white); pdf.setFont("Helvetica-Bold", 18); pdf.drawString(right_x + 20, panel_y + panel_h - 38, "Predicted Label")
        pdf.setFillColor(HexColor("#151a24")); pdf.roundRect(right_x + 20, panel_y + panel_h - 104, right_w - 40, 48, 10, fill=1, stroke=0)
        pdf.setFillColor(label_color); pdf.circle(right_x + 40, panel_y + panel_h - 80, 6, fill=1, stroke=0)
        pdf.setFillColor(white); pdf.setFont("Helvetica-Bold", 18); pdf.drawString(right_x + 54, panel_y + panel_h - 87, str(row["predicted_label"]))
        pdf.drawString(right_x + 20, panel_y + panel_h - 142, "Qwen Explanation")
        pdf.setFillColor(HexColor("#151a24")); pdf.roundRect(right_x + 20, panel_y + 12, right_w - 40, panel_h - 172, 10, fill=1, stroke=0)
        pdf.setFillColor(white); pdf.setFont("Helvetica", 11.5)
        explanation_lines = _wrap_words(str(row["explanation_ar"]), 48)[:4]
        y = panel_y + panel_h - 172
        for line in explanation_lines:
            pdf.drawString(right_x + 34, y, line); y -= 17
        pdf.showPage()
    pdf.save()
    reader = PdfReader(str(destination))
    if len(reader.pages) != len(selected):
        raise ValueError(f"PDF page count mismatch: expected {len(selected)}, got {len(reader.pages)}")


def export_vector_artifacts(selected: pd.DataFrame, output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        write_svg(row, _output_path(root, row, "svg"))
    pdf_path = root / "CAMeLBERT_1NN_Gradio_examples.pdf"
    write_combined_pdf(selected, pdf_path)
    (root / "visual_manifest.json").write_text(
        json.dumps(
            {
                "examples": selected["example_id"].tolist(),
                "semantic_themes": selected[
                    ["example_id", "semantic_theme", "semantic_theme_ar"]
                ].to_dict("records"),
                "svg_count": int(len(selected)),
                "pdf": pdf_path.name,
                "pdf_pages": int(len(selected)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pdf_path
