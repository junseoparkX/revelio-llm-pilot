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

## Selected production prompt: improved_v2

The original three-model trial used `baseline_v1`. After that trial, two revised
prompts were tested with the same 300 postings, the same `gpt-5.6-luna` model,
medium reasoning, frozen reference labels, and the same operational survival rule.
Only the candidate prompt was rerun; the saved baseline was reused.

| Scope and measure | improved_v2 | improved_v2_1 |
|---|---:|---:|
| Overall survival recall | **116/117 (99.15%)** | 113/117 (96.58%) |
| P1 survival recall | **45/45 (100.00%)** | 44/45 (97.78%) |
| P1 with full reference text: survival recall | **41/41 (100.00%)** | 40/41 (97.56%) |
| Overall operational group accuracy | **91.67%** | 90.33% |
| Successful calls | 300/300 | 300/300 |
| Recorded API cost | **USD 0.3024** | USD 0.3166 |

`improved_v2_1` produced stronger evidence-contract rates in some P1 subsets and
higher exact group agreement for `P1_REFERENCE_FULL`. However, it screened out
four reference-relevant postings overall, including one P1 posting. `improved_v2`
screened out one reference-relevant posting overall and none in P1. Because the
larger run is a high-recall first pass, survival recall takes priority over a small
gain in exact agreement or evidence formatting. The selected P1 production prompt
is therefore `improved_v2`, not `improved_v2_1`.

The frozen prompt-and-schema bundle hash is:

```text
abb390da6886d326901544f771cde7f82a94d952f96eb91df3ac4f46d4cf8f2f
```

The exact selected English system prompt below is stored in
`src/revelio_pilot/prompt.py`. `baseline_v1` and the experimental
`improved_v2_1` remain versioned in that file for reproducibility, but P1
explicitly freezes `improved_v2`.

<details>
<summary>Show the complete system prompt</summary>

```text
You are independently reviewing a job posting for a research study of employer demand for SEO and AI-search visibility work. Classify assigned work, not keyword presence.

Use the supplied title and description together.

1. Check text completeness first.
- FULL: the operative duties are present and readable, even if boilerplate is omitted.
- PARTIAL: the text itself is cut off or omits a material duties section, so at least one duty cannot be resolved.
- UNREADABLE: blank, corrupted, or not meaningfully interpretable.
Do not infer truncation from length alone. If missing or contradictory text prevents a decision for either duty, mark that duty UNCERTAIN and make summary_group UNCERTAIN. Do not turn absent evidence in incomplete text into NO.

2. Attribute the work to the worker.
Count only a current action, ownership area, deliverable, or measured outcome assigned to this role. Do not infer a duty from prior experience, qualifications, preferred skills, employer capabilities, product descriptions, page chrome, or a keyword alone. A title can support a duty only when it clearly names the function and the description is compatible; an unexplained acronym in weak or unrelated context is not enough.

3. Judge SEO and GEO independently.
- SEO duty: improve or measure a website's or content's organic visibility, ranking, indexing, discoverability, or traffic in external conventional search engines. Strong cues include organic search, Google/SERP rankings, crawling/indexing, technical SEO, keyword optimization, and SEO performance.
- GEO duty: improve or measure public content's discovery, citation, recommendation, or inclusion in external AI-generated answers or AI-search systems. Strong cues include AI-generated answers, answer engines, LLM discoverability, AI-powered search results, citation/share-of-voice in systems such as ChatGPT or Perplexity, and structuring public content for those systems.
- AEO, GEO, or generative-engine optimization counts as GEO when it is an assigned workstream in an external search/content visibility context. If the acronym's meaning or external target remains materially ambiguous, use UNCERTAIN rather than assuming YES.
- Generic "search," discoverability, personalization, recommendations, or content consumption does not establish SEO or GEO without the relevant external target.

4. Apply exclusions by context.
Do not count internal product search, retrieval, ranking, RAG, recommendation, or model engineering; paid search or SEM alone; marketplace/app-store search alone; AI used only to create content; sales or promotion of SEO/GEO services without ownership of delivery or results; qualification-only mentions; company capabilities; or geographic meanings of GEO. Managing or executing client SEO/GEO delivery can count when the role owns that work or its outcomes.

5. Complete the fields consistently.
- For a YES duty, centrality is PRIMARY or SECONDARY and evidence is a short exact substring that contains the assigned action and relevant target when possible.
- For a NO duty, centrality is NOT_APPLICABLE and evidence is an empty string.
- For an UNCERTAIN duty, centrality is UNCLEAR and uncertainty_reason names the missing or ambiguous fact. Evidence may quote the ambiguous phrase exactly, or be empty when the problem is missing text.
- Evidence must be copied verbatim from the supplied title or description. Do not repair spelling, punctuation, capitalization, whitespace, or HTML entities.
- summary_group is derived from the two duty decisions: BOTH, SEO_ONLY, GEO_ONLY, NEITHER, or UNCERTAIN when either duty is UNCERTAIN.
- required_prior_experience summarizes explicit required or preferred prior work experience; use NO when none is stated. prior_experience_evidence is one exact quote, or an empty string when none is stated.
- seo_background_for_geo is EXPLICIT or SUGGESTIVE only when the posting connects prior SEO background to current GEO work; otherwise NO_EVIDENCE.
- adjacent_type is one concise uppercase exclusion category when useful, such as PAID_SEARCH, INTERNAL_PRODUCT_SEARCH, INTERNAL_AI_PRODUCT, MARKETPLACE_SEARCH, AI_CONTENT_CREATION, SALES_SERVICE, QUALIFICATION_ONLY, PAGE_CHROME, or GEOGRAPHIC_GEO; otherwise use an empty string.

Return only one JSON object matching the supplied schema. concise_rationale must contain exactly two labeled sentences in this order: "SEO: <YES|NO|UNCERTAIN> — <why>. GEO: <YES|NO|UNCERTAIN> — <why>." For YES, name the assigned action and external target. For NO, name the decisive exclusion or state that the complete posting assigns no such external-visibility work. For UNCERTAIN, name the exact ambiguity or missing section. Explain the decision; do not merely repeat the label or evidence quote.
```

</details>

For each of the 300 postings, the following English template supplies the case-specific text:

```text
JOB ID: {job_id}
TITLE RAW: {title_raw}
TRANSLATED TITLE: {jobtitle_translated}

DESCRIPTION:
{description}
```

The JSON field definitions and allowed values are enforced separately through the same structured-output schema for every provider. They are not copied into the prose prompt. The prompt and schema are fingerprinted together; the paid run stops if either one changes after freezing.

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

For this pilot, the 300 saved human reviews are frozen as the reference labels. The metrics measure agreement with this reviewed set; they are not population prevalence estimates. The Excel workbook reports quotation or formatting issues separately without changing the saved duty judgments.

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

The reference decisions are frozen before model inference. Model disagreements become a follow-up inspection queue; they do not overwrite the saved human judgments.

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
  --reference-origin human_reviewed_300 `
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
  --run-id pilot-300-v1 `
  --require-prompt-hash <PROMPT_SHA256>
```

Do not change the prompt after reviewing model results. Full options are available with `python -m revelio_pilot.prepare --help` and `python -m revelio_pilot.run_pilot --help`.

### Savepoints and safe restart

Keep the same `--run-id` when restarting an interrupted run. The runner creates one atomic checkpoint for every posting-model pair under:

```text
outputs/runs/<RUN_ID>/checkpoints/
```

Each checkpoint records the attempt count, received response, token use, estimated cost, validation state, and terminal result. The manifest and usage table are also rewritten atomically as progress is made. On restart, completed pairs are skipped, retry counts and accumulated cost are preserved, and a response that was already received can be validated and committed without another API call. A process lock prevents two copies of the same run from sending duplicate requests. If a shutdown leaves only the final JSONL line incomplete, startup repairs that tail from the durable checkpoint and records the recovery in the manifest; corruption inside a file still stops the run for inspection.

There is one unavoidable edge case: the process can stop while a request is still at the provider, before any response is saved locally. On restart, the runner stops at that pair instead of guessing, because the request may already have been billed. Choose exactly one of the following actions on that first restart.

If a replacement call is intentionally approved, use:

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_reviewed_300.csv `
  --config configs/pilot.yaml `
  --phase frozen_test `
  --run-id pilot-300-v1 `
  --require-prompt-hash <PROMPT_SHA256> `
  --retry-ambiguous-in-flight
```

That option is explicit because it can create one duplicate billable call. The output records the ambiguity instead of hiding it.

If the pair should remain a reported API failure with no possible duplicate charge, use the same command with:

```text
--accept-ambiguous-in-flight-as-failure
```

This records `interrupted_in_flight` in `failures.jsonl` and continues with the remaining pairs without sending that request again.

Create the Excel evaluation:

```powershell
python -m revelio_pilot.evaluate `
  --predictions outputs/runs/<RUN_ID>/predictions.jsonl `
  --failures outputs/runs/<RUN_ID>/failures.jsonl `
  --reference outputs/private_reference/reference_labels.jsonl `
  --source data/pilot_reviewed_300.csv `
  --output outputs/runs/<RUN_ID>/evaluation
```

### Improved-prompt small pilot

The completed Luna-medium baseline was retained rather than rerun. The versioned
`improved_v2` prompt added separate `SEO:` and `GEO:` explanations, tighter external-
visibility and acronym rules, and an evidence/label consistency contract. Both
`improved_v2` and the later experimental `improved_v2_1` were run on the same 300
postings. The comparison above selects `improved_v2` because it achieved the best
overall and P1 survival recall. The comparison tool produces Excel and HTML reports
while separating references marked `FULL` from reference/input-scope mismatches.

See [the prompt v2 small-pilot runbook](docs/prompt_v2_small_pilot.md). The relevant
commands use `--prompt-version improved_v2` and
`python -m revelio_pilot.compare_prompts`.

## Full P1 expansion

After the small-model comparison, the production choice is
`gpt-5.6-luna` with `reasoning_effort: medium`. The full-P1 code reuses the
frozen pilot prompt and validated runner, prepares all 32,972
`P1_EXPLICIT_AEO_GEO` postings as deterministic shards, supports bounded
parallel execution and restart, and refuses to consolidate incomplete or
overlapping results.

See [the P1 Luna-medium runbook](docs/p1_luna_medium_runbook.md). Preparation,
launch, status, and consolidation are exposed through
`python -m revelio_pilot.p1_production` (or `revelio-p1` after installation).

## Data protection

Raw Revelio files, job descriptions, labels, API keys, and run results are ignored by Git. Before calling any provider, confirm that the Revelio agreement and the organization's data policy allow job-posting text to be sent to that provider.

## Current status

The 300-case baseline, `improved_v2`, and experimental `improved_v2_1` runs are
complete. `improved_v2` is frozen for P1 with GPT-5.6 Luna at medium reasoning.
On 2026-09-20, the local preparation produced 32,972 P1 postings in 32 deterministic
shards; the full-input preflight and the 32-shard launcher dry-run both passed with
zero API calls. The generated inputs and all provider outputs remain ignored and
are not included in GitHub. No paid P1 production calls have been started. API
credentials stay only in the local ignored `.env` file and are never documented in
this repository.
