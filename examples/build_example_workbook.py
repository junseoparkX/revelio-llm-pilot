"""Build a small, clearly synthetic example of the Excel deliverable."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from revelio_pilot import evaluate


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    output = repository / "docs" / "example_pilot_results.xlsx"
    with tempfile.TemporaryDirectory() as folder:
        temporary = Path(folder)
        predictions = temporary / "predictions.jsonl"
        references = temporary / "references.jsonl"
        source = temporary / "source.csv"
        fixture = repository / "examples" / "synthetic_records.jsonl"
        rows = fixture.read_text(encoding="utf-8").splitlines()
        predictions.write_text("\n".join(row for row in rows if '"record_type": "prediction"' in row).replace(', "record_type": "prediction"', '') + "\n", encoding="utf-8")
        references.write_text("\n".join(row for row in rows if '"record_type": "reference"' in row).replace(', "record_type": "reference"', '') + "\n", encoding="utf-8")
        pd.DataFrame([
            {"job_id": "EX-001", "title_raw": "Search content manager", "jobtitle_translated": "Search content manager", "description": "Improve organic search rankings and website traffic. Three years of SEO experience required."},
            {"job_id": "EX-002", "title_raw": "AI discovery lead", "jobtitle_translated": "AI discovery lead", "description": "Increase citation frequency in AI-generated answers. SEO experience preferred."},
        ]).to_csv(source, index=False)
        argv = ["evaluate", "--predictions", str(predictions), "--reference", str(references), "--source", str(source), "--output", str(temporary / "evaluation")]
        with patch.object(sys, "argv", argv):
            evaluate.main()
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temporary / "evaluation" / "pilot_results.xlsx", output)
    print(output)


if __name__ == "__main__":
    main()
