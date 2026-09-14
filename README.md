# Job-posting review pilot for SEO and AI search

This project tests whether a small language model can read job postings and consistently identify two kinds of work:

- **SEO work:** improving visibility in regular search engines such as Google.
- **AI-search visibility work:** improving whether a company or its content appears in AI-generated answers. This includes terms such as GEO and AEO.

The model does not write job postings or decide who should be hired. It reads an existing posting and turns the relevant information into a review table that a researcher can check.

## Why run a pilot first?

A keyword search can find a posting that mentions “SEO,” but the mention may only be a qualification. For example:

> Three years of SEO experience preferred.

That sentence describes the applicant's background. It does not prove that SEO is part of the new job. The model is asked to keep current work and requested experience separate.

We first use a small pilot to see whether the instructions work and whether the results are reliable enough for a larger study.

## What the model receives

For each posting, the model sees only:

- the posting ID
- the original title
- the translated title, when available
- the job description

It does not see the researcher's answer, the search group that found the posting, or another model's answer.

## What the model produces

The model returns one structured record per posting. These are the main fields:

| Output | What it means |
|---|---|
| `seo_duty` | Whether SEO is a current duty: `YES`, `NO`, or `UNCERTAIN`. |
| `geo_duty` | Whether AI-search visibility is a current duty: `YES`, `NO`, or `UNCERTAIN`. |
| `seo_centrality` | Whether SEO is a main responsibility or an additional responsibility. |
| `geo_centrality` | Whether AI-search visibility is a main or additional responsibility. |
| `seo_evidence` | Exact words copied from the posting that support the SEO answer. |
| `geo_evidence` | Exact words copied from the posting that support the AI-search answer. |
| `required_prior_experience` | A short summary of experience the employer asks applicants to have. |
| `prior_experience_evidence` | Exact words supporting the experience summary. |
| `seo_background_for_geo` | Whether the posting links an SEO background to current AI-search work. |
| `adjacent_type` | A nearby but excluded category, such as paid search, internal product search, or AI copywriting. |
| `text_completeness` | Whether the available posting text is full, partial, or unreadable. |
| `uncertainty_reason` | Why the model could not make a decision. |
| `summary_group` | The combined result: SEO only, AI-search only, both, neither, or uncertain. |
| `concise_rationale` | A short explanation of the decision. |

The code rejects records with missing fields, impossible combinations, or evidence that cannot be found word-for-word in the posting.

## The Excel result

After evaluation, the main file is:

```text
outputs/runs/<RUN_ID>/evaluation/pilot_results.xlsx
```

A [synthetic example workbook](docs/example_pilot_results.xlsx) is included so the layout can be reviewed without sharing Revelio data.

The workbook is organized for a reader who does not need to know Python:

| Excel tab | What the reader sees |
|---|---|
| **Overview** | One row per model with SEO and AI-search accuracy measures, processing time, possible experience/duty mix-ups, and estimated API cost. |
| **Posting results** | One row for every posting-model pair. It contains the title, model decision, reference decision, exact evidence, explanation, text-quality flag, token use, time, and cost. |
| **Disagreements** | Only the rows where a model differs from the reference review. This is the main human-review queue. |
| **Call quality** | Successful calls, failed calls, invalid outputs, and failure rates for each model. |
| **Confusion matrices** | Counts showing which answers were correct and what each model confused. |
| **Reference quality** | Problems in the benchmark labels themselves, such as missing fields or evidence that is not an exact quotation. |
| **Field guide** | A plain-language definition of every output column and its allowed values. |

The workbook does not include the full job descriptions. It includes short evidence quotations, so it must still be treated as research data and should not be committed to GitHub.

## How the pilot is run

### 1. Use all 300 saved review cases

The pilot uses the same 300 postings summarized in the research report. No new 100-case test is added. Using the entire saved review set avoids throwing away already reviewed examples and keeps every original search group in the model comparison.

The 300 postings have the following composition:

| Review stratum | Cases | Why it is included |
|---|---:|---|
| **P1 — role aligned** | 40 | Strong AI-search signals in a marketing, search, content, or related role. |
| **P1 — technical or unclear** | 45 | AI-search words appear, but the occupation or context makes interpretation harder. |
| **P2 — SEO title or occupation** | 45 | Strong conventional SEO cases without an earlier P1 assignment. |
| **P3 — AI plus relevant work context** | 45 | Broader AI/search context that may or may not describe an actual AI-search duty. |
| **P4 — other related descriptions** | 45 | Broad boundary cases retained to check what P1–P3 miss. |
| **Outside — related role** | 40 | Relevant-looking jobs rejected by the filter; useful for false-negative checks. |
| **Outside — random** | 20 | A broad random negative-control check outside the candidate filter. |
| **Outside — hard negative** | 20 | Keyword-like mentions that should normally remain negative. |
| **Total** | **300** | |

P1 therefore contributes 85 cases and all outside groups together contribute 80. The original archive contained 32,972 P1, 13,188 P2, 4,793 P3, and 121,490 P4 candidate records. A further 110,100 records did not pass the P1–P4 conditions.

The saved review found the following diagnostic rates:

| Original group | Reviewed | SEO duty: YES | AI-search duty: YES |
|---|---:|---:|---:|
| P1 | 85 | 37 (43.5%) | 36 (42.4%) |
| P2 | 45 | 32 (71.1%) | 1 (2.2%) |
| P3 | 45 | 13 (28.9%) | 3 (6.7%) |
| P4 | 45 | 21 (46.7%) | 0 (0.0%) |
| Outside P1–P4 | 80 | 5 (6.3%) | 0 (0.0%) |
| **Total** | **300** | **108 (36.0%)** | **40 (13.3%)** |

These percentages describe this deliberately structured review set. They are useful for comparing where models succeed or fail, but they are **not population prevalence estimates**. P1–P4 and the outside strata were intentionally sampled at different rates. Market-wide shares require either weighting from a probability sample or classification and validation of the larger archive.

The current saved judgments also should not be described as final independent human gold labels without an additional human adjudication step. The Excel workbook therefore reports reference-quality problems separately.

### 2. Freeze the instructions and keep the labels hidden

The model input contains only the posting ID, titles, and description. It excludes the review answer and the P1/P2/P3/P4 assignment. The code records a digital fingerprint of the prompt and output format so the instructions cannot change quietly during the comparison.

### 3. Compare three models on the same 300 postings

Each model reads every posting once. With automatic retries disabled, the planned run contains exactly 900 initial calls:

```text
300 postings × 3 models = 900 calls
```

The current candidates are:

| Provider | Model | Standard input / output price per 1M tokens (USD) |
|---|---|---:|
| [Mistral](https://docs.mistral.ai/models/mistral-small-4-0-26-03) | `mistral-small-2603` | $0.15 / $0.60 |
| [OpenAI](https://developers.openai.com/api/docs/models/gpt-5.6-luna) | `gpt-5.6-luna` | $0.20 / $1.20 |
| [Google](https://ai.google.dev/gemini-api/docs/pricing) | `gemini-3.5-flash-lite` | $0.30 / $2.50 |

### Cost check in Canadian dollars

The estimate below excludes retries. The default configuration sets `max_retries: 0`, limits each response to 1,000 output tokens, and stops new work at USD 6.50.

| Token use per posting and model | Mistral total | OpenAI total | Google total | All 900 calls |
|---|---:|---:|---:|---:|
| 5,000 input + 600 output | USD 0.33 | USD 0.52 | USD 0.90 | **USD 1.75** |
| 10,000 input + 1,000 output | USD 0.63 | USD 0.96 | USD 1.65 | **USD 3.24** |

The locally assembled 300 descriptions average about 5,160 characters and the longest is about 14,983 characters, so the second row is deliberately conservative for ordinary text tokenization. Using the Bank of Canada rate for 11 September 2026, USD 1 = CAD 1.3866, the two scenarios are approximately **CAD 2.43** and **CAD 4.49**. The USD 6.50 hard stop corresponds to about **CAD 9.01** at that rate.

Therefore, the planned 300 × 3 run is expected to remain below CAD 10 **before taxes and card/provider currency-conversion charges**. Prices and the exchange rate must be checked again immediately before the paid run. Failed calls can still be billed; because automatic retries are off, any manual rerun should be budgeted and reported separately.

## How the models are compared

For SEO and AI-search duties, the workbook reports:

- **Precision:** when the model says a duty is present, how often the reference review agrees.
- **Recall:** of the postings where the reference says a duty is present, how many the model found.
- **F1:** a combined precision-and-recall measure.
- **Accuracy:** how often the model and reference give the same answer.
- **Evidence validity:** whether the quoted words appear in the source posting.
- **Failure rate:** how often the provider fails or returns unusable output.
- **Time and cost:** total token use, response time, retries, and estimated API charge.

Model agreement is not treated as truth. Final benchmark labels should be reviewed by people who have not seen the model answers, with disagreements adjudicated before reporting final performance.

## What does not count as SEO or AI-search duty

A posting is not marked positive only because it mentions:

- prior SEO or GEO experience
- a qualification or preferred skill
- paid search or SEM by itself
- internal product search or recommendation systems
- marketplace or app-store search
- AI tools used only to draft content
- the employer's services rather than the worker's responsibilities
- the geographic meaning of “geo”

If the posting is missing or too incomplete, the answer is `UNCERTAIN`, not `NO`.

## Running the project

The project requires Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

Add the provider keys to the local `.env` file. Keys are checked before a run starts.

Prepare all 300 reviewed cases. `data/reviewed_300_input.csv` must contain the posting text and the original `review_stratum`; `data/reviewed_300_labels.csv` must contain the saved answers with matching `job_id` values. These private files are intentionally excluded from Git:

```powershell
python -m revelio_pilot.prepare existing `
  --input data/reviewed_300_input.csv `
  --reference-input data/reviewed_300_labels.csv `
  --reference-origin saved_review_pending_human_adjudication `
  --n 300 --seed 20260914 `
  --output data/pilot_reviewed_300.csv `
  --reference-output outputs/private_reference/reference_labels.jsonl
```

Check that the setup resolves to 300 × 3 = 900 calls without spending money:

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_reviewed_300.csv `
  --config configs/pilot.yaml `
  --phase instruction_check `
  --dry-run
```

The dry run prints the prompt hash. Use that exact value for the paid run:

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_reviewed_300.csv `
  --config configs/pilot.yaml `
  --phase frozen_test `
  --require-prompt-hash <PROMPT_SHA256>
```

Do not change the prompt after reviewing model results. Full options are available with `python -m revelio_pilot.prepare --help` and `python -m revelio_pilot.run_pilot --help`.

Create the Excel evaluation:

```powershell
python -m revelio_pilot.evaluate `
  --predictions outputs/runs/<RUN_ID>/predictions.jsonl `
  --failures outputs/runs/<RUN_ID>/failures.jsonl `
  --reference outputs/private_reference/reference_labels.jsonl `
  --source data/pilot_reviewed_300.csv `
  --output outputs/runs/<RUN_ID>/evaluation
```

## Data protection

Raw Revelio files, job descriptions, labels, API keys, and run results are ignored by Git. Before calling any provider, confirm that the Revelio agreement and the organization's data policy allow job-posting text to be sent to that provider.

## Current status

The code, 300-case preparation, stratum preservation, prompt freeze, validation, cost ceiling, Excel export, and evaluation can be tested without paid API calls. A real 900-call comparison has not been run because API credentials have not been supplied.
