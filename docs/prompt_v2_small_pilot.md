# Luna-medium prompt v2 small-pilot runbook

## What was reviewed

The prompt revision was based on the saved 300-posting outputs, the consolidated
Excel workbook, and the research-report HTML. No baseline API calls need to be
repeated.

The report's substantive definitions are aligned with the study design: classify
current duties, assess SEO and GEO independently, keep prior experience separate,
and require an external visibility target. The main gap is in the model-output
contract. The baseline prompt asks for one short rationale, so it does not reliably
produce a distinct "why SEO" and "why GEO" explanation.

The saved Luna-medium baseline has 299 valid outputs and one failure. It has 28
summary-group disagreements across all 300 cases, but those disagreements should
not all be treated as prompt errors:

- Overall valid-output group accuracy is 90.64%.
- Among the 246 references marked `FULL`, accuracy is 230/246 = 93.50%.
- P1 valid-output accuracy is 71/84 = 84.52%.
- Among the 68 P1 references marked `FULL`, accuracy is 63/68 = 92.65%.
- Eleven of the 28 disagreements occur where the human reference is marked
  `PARTIAL`. Several of those labels describe a 1,200-character cutoff even though
  the inference input contains a fuller description. The comparison therefore
  reports `P1_REFERENCE_FULL` separately and treats it as the primary benchmark.
- Five successful Luna-medium outputs put evidence in a field whose duty was
  `NO`. Prompt v2 makes that contract explicit, and the validator now rejects it.

The HTML still describes the pilot as planned rather than reporting the completed
Luna-medium decision. It is useful for definitions and examples, but it is not the
source of the final model-comparison metrics.

## What prompt v2 changes

- Checks whether operative duties are actually present before using `UNCERTAIN`.
- Requires an assigned action and the relevant external target, not just a keyword.
- Separates conventional organic-search cues from AI-answer/AI-search cues.
- Clarifies AEO/GEO acronym handling, internal search/RAG exclusions, and the
  difference between selling a service and owning delivery.
- Requires exactly two labeled rationale sentences: one for SEO and one for GEO.
- Requires blank evidence for `NO`, exact evidence for `YES`, and a specific reason
  for `UNCERTAIN`.

The output schema is unchanged, so the new run remains directly comparable with
the saved baseline.

## Free preflight

This prints the candidate prompt hash and creates no output files or API calls:

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_reviewed_300.csv `
  --config configs/prompt_v2_luna_medium.yaml `
  --prompt-version improved_v2 `
  --phase instruction_check `
  --dry-run
```

## Candidate-only paid run

Use the hash printed by the dry run. This sends only 300 improved-prompt calls to
Luna medium. It does not rerun the existing baseline.

```powershell
python -m revelio_pilot.run_pilot `
  --input data/pilot_reviewed_300.csv `
  --config configs/prompt_v2_luna_medium.yaml `
  --prompt-version improved_v2 `
  --phase frozen_test `
  --run-id luna-medium-prompt-v2-300 `
  --require-prompt-hash <IMPROVED_V2_SHA256>
```

The config keeps the same 1,000-token limit, zero-retry policy, input rows, seed,
model, and effective medium reasoning as the saved baseline. Its USD 0.50 ceiling
is above the baseline Luna-medium observed cost of about USD 0.2444 but still
prevents an unbounded run.

## Evaluate and compare

```powershell
python -m revelio_pilot.evaluate `
  --predictions outputs/runs/luna-medium-prompt-v2-300/predictions.jsonl `
  --failures outputs/runs/luna-medium-prompt-v2-300/failures.jsonl `
  --reference outputs/private_reference/reference_labels.jsonl `
  --source data/pilot_reviewed_300.csv `
  --output outputs/runs/luna-medium-prompt-v2-300/evaluation

python -m revelio_pilot.compare_prompts `
  --candidate-run outputs/runs/luna-medium-prompt-v2-300 `
  --output outputs/prompt-comparisons/luna-medium-v1-v2
```

The comparison command verifies that the two runs use the same 300 rows, model ID,
effective reasoning level, output-token limit, retry policy, and seed. It then
writes CSV, Excel, JSON metadata, and a standalone HTML comparison. The baseline
path defaults to the existing `outputs/runs/pilot-300-v3-live` run.

Review `P1_REFERENCE_FULL` first, then inspect every `REGRESSED`, `IMPROVED`, and
`CHANGED_OTHER` case. The deterministic rationale score checks that the two
labeled explanations exist and match the returned duty labels; it does not replace
human review of whether the explanation is substantively correct.

## Prepare the existing v2 result for review in Codex with Astra

This is an offline packaging step. It does not call the OpenAI API and does not
read `OPENAI_API_KEY`. It reuses the saved v1 and v2 outputs, summarizes the two
existing Excel workbooks, and selects risk-based cases in this order: survival
misses, P1 duty disagreements, output-contract failures, other duty disagreements,
and agreement controls across strata.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m revelio_pilot.compare_prompts `
  --candidate-run outputs/runs/luna-medium-prompt-v2-300 `
  --output outputs/prompt-comparisons/luna-medium-v1-v2 `
  --codex-astra-review-package
```

The same output folder receives:

- `codex_astra_review_prompt.txt`: the complete English review prompt to use in Codex.
- `codex_astra_review_packet.json`: metrics, workbook structure, and selected source cases.
- `codex_astra_review_metadata.json`: hashes, size, selected-case count, and `api_calls: 0`.

The review prompt tells Astra to prioritize survival recall, verify why each case
is SEO and/or GEO, separate current duties from requested experience, check exact
evidence against the supplied posting text, assess professor-direction alignment,
and distinguish English reader-facing workbook labels from original-language
source or human-reference text.
