from __future__ import annotations

import json
import os
import tempfile
import unittest
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook

from revelio_pilot import providers
from revelio_pilot import evaluate
from revelio_pilot import run_pilot
from revelio_pilot.prepare import _boolean_series, existing
from revelio_pilot.schema import REQUIRED_FIELDS, validate_prediction


def valid_prediction(**updates):
    value = {
        "seo_duty": "YES", "geo_duty": "NO", "seo_centrality": "PRIMARY",
        "geo_centrality": "NOT_APPLICABLE", "seo_evidence": "Improve organic search rankings",
        "geo_evidence": "", "required_prior_experience": "three years",
        "prior_experience_evidence": "three years of SEO experience",
        "seo_background_for_geo": "NO_EVIDENCE", "adjacent_type": "",
        "text_completeness": "FULL", "uncertainty_reason": "",
        "summary_group": "SEO_ONLY", "concise_rationale": "SEO is an assigned duty; GEO is not.",
    }
    value.update(updates)
    return value


class SchemaTests(unittest.TestCase):
    def test_valid_prediction_and_evidence(self):
        source = "Improve organic search rankings; requires three years of SEO experience."
        self.assertEqual(validate_prediction(valid_prediction(), source), [])

    def test_rejects_extra_fields_and_inconsistent_group(self):
        errors = " | ".join(validate_prediction(valid_prediction(summary_group="BOTH", invented=True)))
        self.assertIn("unexpected fields", errors)
        self.assertIn("summary_group must be SEO_ONLY", errors)

    def test_requires_exact_source_quote(self):
        errors = validate_prediction(valid_prediction(), "different source")
        self.assertTrue(any("exact substring" in error for error in errors))

    def test_schema_field_set_is_exact(self):
        self.assertEqual(set(valid_prediction()), set(REQUIRED_FIELDS))


class PrepareTests(unittest.TestCase):
    def test_string_false_is_not_truthy(self):
        result = _boolean_series(pd.Series(["false", "TRUE", "0", "yes"]))
        self.assertEqual(result.tolist(), [False, True, False, True])

    def test_existing_samples_only_reviewed_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidates = pd.DataFrame({
                "job_id": [str(i) for i in range(6)], "title_raw": ["t"] * 6,
                "jobtitle_translated": ["t"] * 6, "description": ["d"] * 6,
            })
            references = pd.DataFrame([
                {"job_id": "1", **valid_prediction()}, {"job_id": "3", **valid_prediction()},
            ])
            candidates.to_csv(root / "candidates.csv", index=False)
            references.to_csv(root / "references.csv", index=False)
            args = Namespace(
                input=str(root / "candidates.csv"), sheet=None, n=2, seed=7,
                reference_input=str(root / "references.csv"), reference_sheet=None,
                reference_col="assistant_final_judgment", reference_origin="test",
                reference_output=str(root / "private" / "labels.jsonl"), output=str(root / "sample.csv"),
            )
            existing(args)
            sampled = pd.read_csv(root / "sample.csv", dtype={"job_id": str})
            self.assertEqual(set(sampled["job_id"]), {"1", "3"})
            records = [json.loads(line) for line in (root / "private" / "labels.jsonl").read_text().splitlines()]
            self.assertEqual({record["job_id"] for record in records}, {"1", "3"})


class ProviderPayloadTests(unittest.TestCase):
    def test_openai_uses_strict_schema_and_no_storage(self):
        body = {"output": [{"content": [{"type": "output_text", "text": "{}"}]}], "usage": {}}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), patch.object(
            providers, "_post", return_value=body
        ) as post:
            providers.call_openai("model", "system", "user")
        payload = post.call_args.kwargs["payload"]
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])

    def test_missing_key_fails_before_http(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(providers, "_post") as post:
            with self.assertRaises(RuntimeError):
                providers.call_mistral("model", "system", "user")
        post.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_evaluator_writes_metrics_and_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prediction = {
                "job_id": "1", "pilot_stratum": "new_random", "model_name": "m",
                "prediction": valid_prediction(), "estimated_cost_usd": 0.01,
                "latency_seconds": 0.2, "attempts": 1,
            }
            reference = {"job_id": "1", "reference_origin": "human_adjudicated", "reference": valid_prediction()}
            (root / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            (root / "reference.jsonl").write_text(json.dumps(reference) + "\n", encoding="utf-8")
            pd.DataFrame([{
                "job_id": "1", "title_raw": "", "jobtitle_translated": "",
                "description": "Improve organic search rankings; three years of SEO experience",
            }]).to_csv(root / "source.csv", index=False)
            argv = [
                "evaluate", "--predictions", str(root / "predictions.jsonl"),
                "--reference", str(root / "reference.jsonl"), "--source", str(root / "source.csv"),
                "--output", str(root / "evaluation"),
            ]
            with patch.object(sys, "argv", argv):
                evaluate.main()
            metrics = pd.read_csv(root / "evaluation" / "per_model_metrics.csv")
            self.assertEqual(metrics.loc[0, "group_accuracy"], 1.0)
            self.assertTrue((root / "evaluation" / "confusion_matrices.csv").exists())
            quality = pd.read_csv(root / "evaluation" / "reference_quality.csv")
            self.assertTrue(quality.loc[0, "schema_or_logic_valid"])
            workbook = load_workbook(root / "evaluation" / "pilot_results.xlsx", data_only=False)
            self.assertEqual(
                workbook.sheetnames,
                ["Overview", "Posting results", "Disagreements", "Call quality", "Confusion matrices", "Reference quality", "Field guide"],
            )
            self.assertEqual(workbook["Posting results"]["A2"].value, "1")
            self.assertEqual(workbook["Field guide"]["A2"].value, "seo_duty")


class RunnerIntegrationTests(unittest.TestCase):
    def test_mock_provider_run_checkpoints_validated_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame([{
                "job_id": "1", "title_raw": "SEO role", "jobtitle_translated": "SEO role",
                "description": "Improve organic search rankings; three years of SEO experience",
                "pilot_stratum": "existing_instruction_check",
            }]).to_csv(root / "input.csv", index=False)
            (root / "config.yaml").write_text(
                "seed: 7\nmax_retries: 1\ntimeout_seconds: 10\nmax_cost_usd: 1\nmodels:\n"
                "  - name: mock_openai\n    provider: openai\n    model_id: test-model\n"
                "    input_usd_per_million: 0.2\n    output_usd_per_million: 1.2\n",
                encoding="utf-8",
            )
            argv = [
                "run", "--input", "input.csv", "--config", "config.yaml",
                "--phase", "instruction_check", "--run-id", "test-run",
            ]
            prior = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys, "argv", argv), patch.dict(
                    os.environ, {"OPENAI_API_KEY": "test"}
                ), patch.object(
                    run_pilot, "call",
                    side_effect=[
                        ("not-json", {"input_tokens": 100, "output_tokens": 50}, {"id": "mock-1"}, 0.01),
                        (json.dumps(valid_prediction()), {"input_tokens": 100, "output_tokens": 50}, {"id": "mock-2"}, 0.01),
                    ],
                ):
                    run_pilot.main()
            finally:
                os.chdir(prior)
            manifest = json.loads((root / "outputs" / "runs" / "test-run" / "manifest.json").read_text())
            self.assertEqual(manifest["successful_calls"], 1)
            self.assertFalse(manifest["stopped_for_budget"])
            raw_lines = (root / "outputs" / "runs" / "test-run" / "raw_responses.jsonl").read_text().splitlines()
            self.assertEqual(len(raw_lines), 2)
            self.assertAlmostEqual(manifest["estimated_cost_usd"], 0.00016)
            result = json.loads((root / "outputs" / "runs" / "test-run" / "predictions.jsonl").read_text())
            self.assertEqual(result["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
