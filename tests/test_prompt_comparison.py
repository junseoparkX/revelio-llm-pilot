from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from revelio_pilot.compare_prompts import (
    ASTRA_AUDIT_SYSTEM_PROMPT,
    build_astra_audit_user_prompt,
    effective_reasoning,
    evidence_contract,
    rationale_contract,
    select_astra_audit_cases,
    validate_comparability,
    write_html,
    write_workbook,
    summary_frame,
)
from revelio_pilot.prompt import get_system_prompt, prompt_hash


def prediction(**updates):
    value = {
        "seo_duty": "YES",
        "geo_duty": "NO",
        "seo_centrality": "PRIMARY",
        "geo_centrality": "NOT_APPLICABLE",
        "seo_evidence": "Improve organic search rankings",
        "geo_evidence": "",
        "required_prior_experience": "NO",
        "prior_experience_evidence": "",
        "seo_background_for_geo": "NO_EVIDENCE",
        "adjacent_type": "",
        "text_completeness": "FULL",
        "uncertainty_reason": "",
        "summary_group": "SEO_ONLY",
        "concise_rationale": (
            "SEO: YES — The role improves external organic rankings. "
            "GEO: NO — The complete posting assigns no AI-search visibility work."
        ),
    }
    value.update(updates)
    return value


class PromptVersionTests(unittest.TestCase):
    def test_baseline_hash_is_unchanged(self):
        self.assertEqual(
            prompt_hash(get_system_prompt("baseline_v1")),
            "adc3fa224387c812910a16578af240088b3beef1e7aaacd7cc21853ed7f207db",
        )
        self.assertNotEqual(
            prompt_hash(get_system_prompt("baseline_v1")),
            prompt_hash(get_system_prompt("improved_v2")),
        )
        self.assertNotEqual(
            prompt_hash(get_system_prompt("improved_v2")),
            prompt_hash(get_system_prompt("improved_v2_1")),
        )

    def test_rationale_and_evidence_contracts(self):
        value = prediction()
        source = "Improve organic search rankings"
        self.assertTrue(rationale_contract(value))
        self.assertTrue(evidence_contract(value, source))
        self.assertFalse(
            rationale_contract(value | {"concise_rationale": "SEO is present; GEO is absent."})
        )
        self.assertFalse(evidence_contract(value | {"geo_evidence": "AI search"}, source))

    def test_default_gpt_5_6_reasoning_matches_explicit_medium(self):
        baseline_model = {"provider": "openai", "model_id": "gpt-5.6-luna"}
        candidate_model = baseline_model | {"reasoning_effort": "medium"}
        self.assertEqual(effective_reasoning(baseline_model), "medium")
        self.assertEqual(effective_reasoning(candidate_model), "medium")

    def test_survival_recall_retains_uncertain_and_failures(self):
        cases = pd.DataFrame(
            [
                {
                    "target_bucket": "P1", "pilot_stratum": "P1_ROLE_ALIGNED",
                    "reference_text_completeness": "FULL", "reference_seo_duty": "YES",
                    "reference_geo_duty": "NO", "reference_summary_group": "SEO_ONLY",
                    "baseline_status": "SUCCESS", "baseline_seo_duty": "NO",
                    "baseline_geo_duty": "NO", "baseline_summary_group": "NEITHER",
                    "candidate_status": "SUCCESS", "candidate_seo_duty": "UNCERTAIN",
                    "candidate_geo_duty": "NO", "candidate_summary_group": "UNCERTAIN",
                },
                {
                    "target_bucket": "P1", "pilot_stratum": "P1_ROLE_ALIGNED",
                    "reference_text_completeness": "FULL", "reference_seo_duty": "NO",
                    "reference_geo_duty": "YES", "reference_summary_group": "GEO_ONLY",
                    "baseline_status": "FAILURE", "baseline_seo_duty": "",
                    "baseline_geo_duty": "", "baseline_summary_group": "",
                    "candidate_status": "SUCCESS", "candidate_seo_duty": "NO",
                    "candidate_geo_duty": "YES", "candidate_summary_group": "GEO_ONLY",
                },
            ]
        )
        for prefix in ("baseline", "candidate"):
            cases[f"{prefix}_rationale_contract"] = False
            cases[f"{prefix}_evidence_contract"] = False
            cases[f"{prefix}_structural_contract"] = False
            cases[f"{prefix}_input_tokens"] = 0
            cases[f"{prefix}_output_tokens"] = 0
            cases[f"{prefix}_estimated_cost_usd"] = 0.0
        summary = summary_frame(cases, "baseline_v1", "improved_v2_1")
        all_candidate = summary[
            (summary["segment"] == "ALL") & (summary["version"] == "improved_v2_1")
        ].iloc[0]
        all_baseline = summary[
            (summary["segment"] == "ALL") & (summary["version"] == "baseline_v1")
        ].iloc[0]
        self.assertEqual(all_candidate["survival_recall"], 1.0)
        self.assertEqual(all_baseline["survival_recall"], 0.5)

    def test_astra_audit_prioritizes_survival_miss_and_p1_disagreement(self):
        common = {
            "candidate_status": "SUCCESS",
            "candidate_rationale_contract": True,
            "candidate_evidence_contract": True,
            "candidate_structural_contract": True,
            "candidate_group_correct": False,
        }
        cases = pd.DataFrame(
            [
                common | {
                    "job_id": "3", "pilot_stratum": "P2", "target_bucket": "P2",
                    "reference_seo_duty": "YES", "reference_geo_duty": "NO",
                    "candidate_seo_duty": "NO", "candidate_geo_duty": "NO",
                },
                common | {
                    "job_id": "2", "pilot_stratum": "P1_ROLE_ALIGNED", "target_bucket": "P1",
                    "reference_seo_duty": "NO", "reference_geo_duty": "YES",
                    "candidate_seo_duty": "NO", "candidate_geo_duty": "UNCERTAIN",
                },
                common | {
                    "job_id": "1", "pilot_stratum": "P4", "target_bucket": "P4",
                    "reference_seo_duty": "NO", "reference_geo_duty": "NO",
                    "candidate_seo_duty": "YES", "candidate_geo_duty": "NO",
                },
            ]
        )
        selected = select_astra_audit_cases(cases, 3)
        self.assertEqual(selected[0]["row"]["job_id"], "3")
        self.assertEqual(selected[0]["reasons"], ["SURVIVAL_MISS"])
        self.assertEqual(selected[1]["row"]["job_id"], "2")
        self.assertEqual(selected[1]["reasons"], ["P1_DUTY_DISAGREEMENT"])

    def test_astra_prompt_is_embedded_in_code_for_codex_review(self):
        user_prompt = build_astra_audit_user_prompt({"selected_cases": []})
        self.assertIn("survival-recall logic", user_prompt)
        self.assertIn("EXPECTED RESPONSE SCHEMA", user_prompt)
        self.assertIn("GitHub readiness", user_prompt)
        self.assertIn("Only a successful conclusive SEO=NO", ASTRA_AUDIT_SYSTEM_PROMPT)
        self.assertIn("Write every output field in English", ASTRA_AUDIT_SYSTEM_PROMPT)
        self.assertIn("zero model API calls", ASTRA_AUDIT_SYSTEM_PROMPT)

    def test_comparison_rejects_nonisolated_runs(self):
        common = {
            "selected_jobs_sha256": "jobs",
            "input_sha256": "input",
            "selected_rows": 300,
            "max_output_tokens": 1000,
            "max_retries": 0,
            "seed": 20260914,
        }
        baseline = common | {
            "prompt_sha256": prompt_hash(get_system_prompt("baseline_v1")),
            "models": [
                {"name": "gpt_5_6_luna", "provider": "openai", "model_id": "gpt-5.6-luna"}
            ],
        }
        candidate = common | {
            "prompt_sha256": prompt_hash(get_system_prompt("improved_v2")),
            "models": [
                {
                    "name": "gpt_5_6_luna_prompt_v2",
                    "provider": "openai",
                    "model_id": "gpt-5.6-luna",
                    "reasoning_effort": "medium",
                }
            ],
        }
        checks = validate_comparability(
            baseline, candidate, "gpt_5_6_luna", "gpt_5_6_luna_prompt_v2"
        )
        self.assertIn("same selected jobs", checks)
        with self.assertRaises(ValueError):
            validate_comparability(
                baseline,
                candidate | {"selected_jobs_sha256": "different"},
                "gpt_5_6_luna",
                "gpt_5_6_luna_prompt_v2",
            )

    def test_comparison_writers_create_excel_and_html(self):
        summary = pd.DataFrame(
            [{
                "segment": "ALL", "version": "baseline_v1", "assigned_cases": 1,
                "valid_outputs": 1, "failures_or_missing": 0, "success_rate": 1.0,
                "group_accuracy_valid": 1.0, "group_accuracy_operational": 1.0,
                "seo_exact_agreement": 1.0, "geo_exact_agreement": 1.0,
                "seo_f1": 1.0, "geo_f1": 0.0, "rationale_contract_rate": 0.0,
                "evidence_contract_rate": 1.0, "structural_contract_rate": 1.0,
                "input_tokens": 10, "output_tokens": 5, "estimated_cost_usd": 0.001,
            }]
        )
        cases = pd.DataFrame(
            [{
                "comparison_outcome": "CHANGED_OTHER", "job_id": "1",
                "pilot_stratum": "P1_ROLE_ALIGNED", "reference_text_completeness": "FULL",
                "reference_summary_group": "SEO_ONLY", "baseline_summary_group": "SEO_ONLY",
                "candidate_summary_group": "BOTH", "baseline_concise_rationale": "baseline",
                "candidate_concise_rationale": "candidate",
            }]
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_workbook(root / "comparison.xlsx", summary, cases)
            write_html(root / "comparison.html", summary, cases)
            self.assertTrue((root / "comparison.xlsx").exists())
            self.assertIn("Luna Medium prompt comparison", (root / "comparison.html").read_text())


if __name__ == "__main__":
    unittest.main()
