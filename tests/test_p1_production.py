from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from revelio_pilot import p1_production


class P1ProductionTests(unittest.TestCase):
    def test_tier2_plan_uses_all_32_prepared_shards_with_headroom(self):
        plan = p1_production.tier2_parallel_plan(32, 32)
        self.assertEqual(plan["recommended_cap"], 32)
        self.assertGreaterEqual(plan["model_safe_cap"], 32)
        self.assertLess(plan["projected_rpm"], p1_production.TIER2_REQUESTS_PER_MINUTE)
        self.assertLess(plan["projected_total_tpm"], p1_production.TIER2_TOKENS_PER_MINUTE)
        self.assertAlmostEqual(plan["projected_minutes"], 72.9, places=1)

    def test_tier2_plan_rejects_non_positive_parallelism(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            p1_production.tier2_parallel_plan(0, 32)

    def test_launch_parser_defaults_to_all_shards_with_stagger(self):
        args = p1_production.build_parser().parse_args(
            ["launch", "--prompt-hash", "abc123", "--dry-run"]
        )
        self.assertEqual(args.max_parallel, 32)
        self.assertEqual(args.start_interval_seconds, 2.0)

    def test_stable_shard_is_repeatable_and_bounded(self):
        first = [p1_production.stable_shard(str(value), 32) for value in range(100)]
        second = [p1_production.stable_shard(str(value), 32) for value in range(100)]
        self.assertEqual(first, second)
        self.assertTrue(all(0 <= value < 32 for value in first))
        self.assertGreater(len(set(first)), 20)

    def test_load_config_requires_explicit_luna_medium(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(
                "max_retries: 0\ntimeout_seconds: 90\nmax_output_tokens: 2500\n"
                "max_cost_usd: 2\nmodels:\n"
                "  - name: luna\n    provider: openai\n    model_id: gpt-5.6-luna\n"
                "    reasoning_effort: high\n    input_usd_per_million: 0.2\n"
                "    output_usd_per_million: 1.2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "reasoning_effort='medium'"):
                p1_production.load_config(path)

    def test_prepare_filters_p1_and_creates_exact_once_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "candidates.parquet"
            output = root / "prepared"
            pd.DataFrame(
                [
                    {
                        "job_id": index,
                        "title_raw": f"title {index}",
                        "jobtitle_translated": "",
                        "description": f"description {index}",
                        "filter_priority": (
                            p1_production.P1 if index < 7 else "P2_STRONG_SEO"
                        ),
                    }
                    for index in range(10)
                ]
            ).to_parquet(source, index=False)
            args = Namespace(source=str(source), output_dir=str(output), expected_rows=7, shards=3)
            p1_production.prepare(args)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rows"], 7)
            self.assertEqual(sum(item["rows"] for item in manifest["shards"]), 7)
            all_ids = []
            for item in manifest["shards"]:
                shard = pd.read_csv(output / item["file"], dtype={"job_id": str})
                all_ids.extend(shard["job_id"].tolist())
                if not shard.empty:
                    self.assertEqual(set(shard["pilot_stratum"]), {p1_production.P1})
            self.assertEqual(sorted(all_ids), [str(value) for value in range(7)])
            self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_command_for_shard_freezes_prompt_and_run_id(self):
        command = p1_production.command_for_shard(
            input_path=Path("input.csv"),
            config_path=Path("config.yaml"),
            prompt_sha256="abc123",
            shard_run_id="p1-shard-001",
            retry_ambiguous=False,
            accept_ambiguous=False,
        )
        self.assertIn("frozen_test", command)
        prompt_flag = command.index("--prompt-version")
        self.assertEqual(command[prompt_flag + 1], "improved_v2")
        self.assertIn("abc123", command)
        self.assertIn("p1-shard-001", command)
        self.assertNotIn("--retry-ambiguous-in-flight", command)


if __name__ == "__main__":
    unittest.main()
