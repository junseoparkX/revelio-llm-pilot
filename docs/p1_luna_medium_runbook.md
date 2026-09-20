# Full P1 expansion with GPT-5.6 Luna at medium reasoning

This workflow expands the selected frozen `improved_v2` small-pilot prompt to
every posting in `P1_EXPLICIT_AEO_GEO`. It intentionally keeps that prompt and
the prediction schema unchanged. The model sees only the job ID, titles, and
description; it does not see `filter_priority`, human judgments, or pilot
outcomes.

The production config is `configs/p1_luna_medium.yaml`. It fixes the selected
model and reasoning setting explicitly:

```text
model_id: gpt-5.6-luna
reasoning_effort: medium
prompt_version: improved_v2
```

The code uses the already-tested synchronous runner rather than introducing a
second inference implementation. Each shard therefore inherits strict
structured output, evidence checks, per-call checkpoints, prompt fingerprints,
no automatic retries, and conservative handling of interrupted in-flight calls.

## 1. Prepare the full P1 cohort

From the `revelio-llm-pilot` directory:

```powershell
python -m revelio_pilot.p1_production prepare
```

The preparation step expects exactly 32,972 P1 postings from
`../data/interim/high_potential_llm_candidates.parquet`. It stops if that count
has changed, a job ID is missing or duplicated, or the destination already
contains files. It writes one master input, 32 deterministic shards, and a
hash manifest under `data/p1_luna_medium/`.

## 2. Freeze-check without API calls

Use the master input for one offline check:

```powershell
python -m revelio_pilot.run_pilot `
  --input data/p1_luna_medium/p1_all.csv `
  --config configs/p1_luna_medium.yaml `
  --phase instruction_check `
  --prompt-version improved_v2 `
  --dry-run
```

Record the printed prompt hash. No output file or API call is created by that
command.

The launcher itself can also be inspected without starting child processes:

```powershell
python -m revelio_pilot.p1_production launch `
  --prompt-hash <PROMPT_SHA256> `
  --max-parallel 32 `
  --dry-run
```

## 3. Paid run (only when separately approved)

```powershell
python -m revelio_pilot.p1_production launch `
  --prompt-hash <PROMPT_SHA256> `
  --max-parallel 32
```

The 32 prepared shards make 32 simultaneous shard processes the practical
maximum for this run. It is also the default, so `--max-parallel 32` may be
omitted. A two-second process-start stagger is enabled by default to soften the
initial traffic ramp without materially changing completion time.

OpenAI's published GPT-5.6 Luna Tier 2 limits are 5,000 requests/minute and
2,000,000 tokens/minute. Across the completed 300-case `improved_v2` pilot, a
call averaged 2,268.59 input tokens, 461.97 output tokens, and 4.2467 seconds.
At 32 workers, the projected steady load is approximately 452 requests/minute
and 1.235 million observed tokens/minute: about 9% and 62% of the respective
Tier 2 limits. The no-overhead runtime projection is about 73 minutes, compared
with about 292 minutes at the previous eight-worker default. These are planning
estimates, not a throughput guarantee; organization/project traffic shares the
same limits. The launcher refuses settings above either the prepared shard count
or the 80%-utilization safety cap calculated from the frozen v2 pilot.

Official references: [GPT-5.6 Luna model and rate limits](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [rate-limit behavior](https://developers.openai.com/api/docs/guides/rate-limits).

`--max-parallel` controls simultaneous shard processes. The config's USD 2.00
ceiling applies independently to each of the 32 shards, so the aggregate hard
stop is USD 64.00. This is a safety ceiling, not a forecast. Re-check current
pricing, account rate limits, and permission to send Revelio text immediately
before a paid run.

Keep the same run prefix when restarting. Completed posting-model pairs are
skipped. If a process stopped while a request was in flight, the launcher will
not guess whether it was billed. Review that shard and then explicitly choose
either `--retry-ambiguous-in-flight` or
`--accept-ambiguous-in-flight-as-failure`.

## 4. Inspect progress

```powershell
python -m revelio_pilot.p1_production status
```

## 5. Consolidate only after every shard completes

```powershell
python -m revelio_pilot.p1_production consolidate
```

Consolidation refuses partial runs, prompt/model mismatches, duplicates,
missing job IDs, or unexpected job IDs. It writes:

- `outputs/p1_luna_medium/predictions.jsonl`
- `outputs/p1_luna_medium/failures.jsonl`
- `outputs/p1_luna_medium/p1_results.csv`
- `outputs/p1_luna_medium/p1_failures.csv`
- `outputs/p1_luna_medium/summary.json`

These are production model classifications, not human gold labels. All
prepared inputs, responses, and consolidated files remain excluded from Git.
