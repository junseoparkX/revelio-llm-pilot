# Revelio LLM Extraction Pilot

A data-free, reproducible pilot for comparing low-cost language models on SEO versus AI-search visibility (GEO/AEO) duty extraction from job postings.

## Research design

1. Use **50 previously reviewed postings** to test the instructions. These cases are for prompt development, not an unbiased final score.
2. Record the prompt SHA-256 hash and stop changing the instructions.
3. Run the frozen prompt on **100 previously unseen postings**: 50 random candidates and 50 difficult/boundary cases.
4. Compare three models on identical inputs: **150 postings × 3 models = 450 initial calls**.
5. Report the random and difficult strata separately. Treat text-supported reference review—not model agreement—as the benchmark.

The proposed spending ceiling in the research report is **CAD 10**. The default config adds a conservative **USD 7.00 hard stop** because provider bills are denominated in USD; exchange rates, tax, retries, and account funding can differ.

## What is measured

- SEO and GEO precision, recall, F1, accuracy, and confusion matrices
- combined `SEO_ONLY` / `GEO_ONLY` / `BOTH` / `NEITHER` / `UNCERTAIN` accuracy
- literal source-substring validity of returned evidence
- terminal API failure and invalid-output rates
- review flags for possible duty-versus-prior-experience confusion
- latency, attempts, token usage, and estimated API cost

The duty/prior-experience flag is deliberately a proxy requiring human review. Literal quotation validity proves only that the text exists, not that it supports the model's conclusion.

## Privacy boundary

This repository contains no Revelio records or reference judgments. `.gitignore` excludes data files, API keys, and run outputs. Before using any provider, confirm that the Revelio license and the organization's data policy allow sending job-posting text to that provider. The code sends only title/description fields; it never sends screening groups, reference labels, or earlier judgments.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

Add only the provider keys you intend to use to `.env`. The runner checks all configured credentials before creating a run or making a call.

## 1. Prepare 50 instruction-check cases

The input and reference labels may be separate files joined by `job_id`:

```powershell
python -m revelio_pilot.prepare existing `
  --input data/llm_extraction_pilot_300_input.csv `
  --reference-input data/review_labels.csv `
  --reference-origin ai_provisional_error_discovery `
  --n 50 --seed 20260914 `
  --output data/pilot_existing_50.csv `
  --reference-output outputs/private_reference/reference_labels.jsonl
```

Only rows present in the reference file are eligible. Sampling is stratified by `summary_group` when available. The API input is blind; references are written separately under the ignored `outputs/` directory. Use an accurate `--reference-origin` value—do not label AI-provisional judgments as human gold.

## 2. Validate before spending

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_existing_50.csv `
  --config configs/pilot.yaml `
  --phase instruction_check `
  --dry-run
```

The dry run validates the input, model IDs, prompt-freeze rules, and expected call count without requiring keys or creating output files.

## 3. Run the instruction check and freeze the prompt

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_existing_50.csv `
  --config configs/pilot.yaml `
  --phase instruction_check
```

Inspect disagreements, invalid output, and evidence failures. Revise only during this phase. Record the prompt hash printed by the runner after the instructions are final.

## 4. Prepare the unseen 100-case test

Create the fresh candidate pool before examining its labels. It must include a pre-specified boolean `difficult_case` column:

```powershell
python -m revelio_pilot.prepare new `
  --input data/new_candidate_pool.parquet `
  --random-n 50 --difficult-n 50 --seed 20260914 `
  --output data/pilot_new_100.csv
```

Define and document the difficult-case rule before sampling. Do not tune the prompt on these 100 cases.

## 5. Run the frozen test

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_new_100.csv `
  --config configs/pilot.yaml `
  --phase frozen_test `
  --require-prompt-hash <HASH_FROM_STEP_3>
```

The runner requires the hash, uses provider-native structured output, validates exact fields and logical consistency, checks quotations against the supplied posting text, checkpoints every response, supports safe resume with `--run-id`, and stops before the next call when the configured USD ceiling has been reached.

## 6. Evaluate

```powershell
python -m revelio_pilot.evaluate `
  --predictions outputs/runs/<RUN_ID>/predictions.jsonl `
  --failures outputs/runs/<RUN_ID>/failures.jsonl `
  --reference outputs/private_reference/reference_labels.jsonl `
  --source data/pilot_new_100.csv `
  --output outputs/runs/<RUN_ID>/evaluation
```

Evaluation writes `summary.csv`, `per_model_metrics.csv`, `per_stratum_metrics.csv`, `confusion_matrices.csv`, `disagreements.csv`, `call_quality.csv`, `reference_quality.csv`, and `evaluation.json`. Repair or adjudicate reference rows flagged by `reference_quality.csv` before treating scores as benchmark estimates.

## Output and reproducibility

Each run stores a manifest with the input, prompt, and schema hashes; exact model IDs; config; timestamps; expected/success/failure counts; and cost. Raw provider responses are checkpointed separately from validated predictions. Run artifacts remain local because they may contain posting content or judgments.

Pinned model IDs and standard prices were verified on 2026-09-14:

| Provider | Model ID | Input / output per 1M tokens (USD) |
|---|---|---:|
| [Mistral](https://docs.mistral.ai/models/mistral-small-4-0-26-03) | `mistral-small-2603` | $0.15 / $0.60 |
| [OpenAI](https://developers.openai.com/api/docs/models/gpt-5.6-luna) | `gpt-5.6-luna` | $0.20 / $1.20 |
| [Google](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) | `gemini-3.5-flash-lite` | $0.30 / $2.50 |

Confirm availability and pricing in the specific provider account immediately before a paid run.

## Repository layout

```text
configs/pilot.yaml        pinned models, prices, retries, timeout, cost stop
data/README.md            local input contract; all other data ignored
outputs/.gitkeep          local run root; all run contents ignored
src/revelio_pilot/        sampling, prompt, schemas, providers, runner, evaluation
tests/                    offline unit and integration tests
```

## Current status

The offline pipeline, schema validation, sampling, dry run, failure preflight, and evaluator can be tested without API keys. A real provider comparison has **not** been run until credentials are supplied and data-transfer permission is confirmed.
