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
  --max-parallel 8 `
  --dry-run
```

## 3. Paid run (only when separately approved)

```powershell
python -m revelio_pilot.p1_production launch `
  --prompt-hash <PROMPT_SHA256> `
  --max-parallel 8
```

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
