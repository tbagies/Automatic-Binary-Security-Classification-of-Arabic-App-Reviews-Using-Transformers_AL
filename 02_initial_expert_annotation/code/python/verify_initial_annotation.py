"""Verify the expert workbook and its balanced 300-row seed dataset."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def verify(workbook: Path, balanced_csv: Path) -> dict[str, int]:
    security = pd.read_excel(workbook, sheet_name="Security", header=1)
    nonsecurity = pd.read_excel(workbook, sheet_name="Non-security", header=1)
    if len(security) != 166 or len(nonsecurity) != 154:
        raise AssertionError("expert workbook must contain 166 security and 154 non-security rows")
    balanced = pd.read_csv(balanced_csv)
    if len(balanced) != 300:
        raise AssertionError("balanced seed must contain 300 rows")
    if set(balanced["Keyword"].value_counts().tolist()) != {150}:
        raise AssertionError("balanced seed must contain 150 rows per class")
    if balanced[["app_name", "store"]].isna().any().any():
        raise AssertionError("application/store provenance is incomplete")
    return {"expert_security": 166, "expert_nonsecurity": 154, "balanced_seed": 300}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("balanced_csv", type=Path)
    args = parser.parse_args()
    print(verify(args.workbook, args.balanced_csv))


if __name__ == "__main__":
    main()

