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

### 1. Check the instructions on 50 existing cases

Fifty previously reviewed postings are used to find unclear instructions and common mistakes. These results are for prompt development, not final performance estimates.

### 2. Freeze the instructions

The code records a digital fingerprint of the prompt and output format. If either changes, the final test will stop. This prevents quiet changes after seeing the test cases.

### 3. Test 100 new cases

The final test contains:

- 50 randomly selected candidate postings
- 50 difficult or borderline postings

The two groups are reported separately. This prevents a deliberately difficult sample from being mistaken for the normal error rate.

### 4. Compare three models

Each model reads the same 150 postings. The full pilot therefore contains 450 initial calls:

```text
150 postings × 3 models = 450 calls
```

The current candidates are:

| Provider | Model | Standard input / output price per 1M tokens (USD) |
|---|---|---:|
| [Mistral](https://docs.mistral.ai/models/mistral-small-4-0-26-03) | `mistral-small-2603` | $0.15 / $0.60 |
| [OpenAI](https://developers.openai.com/api/docs/models/gpt-5.6-luna) | `gpt-5.6-luna` | $0.20 / $1.20 |
| [Google](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) | `gemini-3.5-flash-lite` | $0.30 / $2.50 |

The research report proposes a CAD 10 ceiling. The config uses a USD 7.00 stop as a conservative operating limit. Prices, exchange rates, taxes, and account access should be checked again immediately before a paid run.

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

Prepare the 50 instruction-check cases:

```powershell
python -m revelio_pilot.prepare existing `
  --input data/llm_extraction_pilot_300_input.csv `
  --reference-input data/review_labels.csv `
  --reference-origin ai_provisional_error_discovery `
  --n 50 --seed 20260914 `
  --output data/pilot_existing_50.csv `
  --reference-output outputs/private_reference/reference_labels.jsonl
```

Check the setup without spending money:

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_existing_50.csv `
  --config configs/pilot.yaml `
  --phase instruction_check `
  --dry-run
```

Run the models by removing `--dry-run`. After the prompt has been reviewed, prepare the new 100 cases and run the final phase with the recorded prompt hash. Full commands are available with `python -m revelio_pilot.prepare --help` and `python -m revelio_pilot.run_pilot --help`.

Create the Excel evaluation:

```powershell
python -m revelio_pilot.evaluate `
  --predictions outputs/runs/<RUN_ID>/predictions.jsonl `
  --failures outputs/runs/<RUN_ID>/failures.jsonl `
  --reference outputs/private_reference/reference_labels.jsonl `
  --source data/pilot_new_100.csv `
  --output outputs/runs/<RUN_ID>/evaluation
```

## Data protection

Raw Revelio files, job descriptions, labels, API keys, and run results are ignored by Git. Before calling any provider, confirm that the Revelio agreement and the organization's data policy allow job-posting text to be sent to that provider.

## Current status

The code, sampling, prompt freeze, validation, retry accounting, Excel export, and evaluation can be tested without paid API calls. A real three-model comparison has not been run because API credentials have not been supplied.
