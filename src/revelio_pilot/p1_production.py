"""Prepare, launch, inspect, and consolidate the full P1 Luna-medium run.

The module deliberately delegates individual API calls to ``run_pilot`` so the
production expansion inherits the pilot's prompt freeze, schema validation,
cost accounting, atomic checkpoints, and conservative restart behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .prompt import get_system_prompt, prompt_hash
from .run_pilot import INPUT_COLUMNS, load_jsonl, validate_config
from .schema import PREDICTION_SCHEMA, REQUIRED_FIELDS


P1 = "P1_EXPLICIT_AEO_GEO"
P1_PROMPT_VERSION = "improved_v2"
P1_PROMPT_SHA256 = prompt_hash(get_system_prompt(P1_PROMPT_VERSION))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = "../data/interim/high_potential_llm_candidates.parquet"
DEFAULT_PREPARED_DIR = "data/p1_luna_medium"
DEFAULT_CONFIG = "configs/p1_luna_medium.yaml"
DEFAULT_RUN_PREFIX = "p1-luna-medium"
DEFAULT_EXPECTED_ROWS = 32_972
DEFAULT_SHARDS = 32
DEFAULT_MAX_PARALLEL = 32
DEFAULT_START_INTERVAL_SECONDS = 2.0
PREPARATION_VERSION = 1
CONSOLIDATION_VERSION = 1
SOURCE_COLUMNS = ["job_id", "title_raw", "jobtitle_translated", "description", "filter_priority"]

# GPT-5.6 Luna Tier 2 limits published by OpenAI on 2026-09-20.
# The utilization target preserves headroom for latency/token variation and for
# other traffic sharing the same organization/project limits.
TIER2_REQUESTS_PER_MINUTE = 5_000
TIER2_TOKENS_PER_MINUTE = 2_000_000
TIER2_TARGET_UTILIZATION = 0.80

# Observed across all 300 successful improved_v2 Luna-medium pilot calls.
V2_MEAN_INPUT_TOKENS = 2_268.586667
V2_MEAN_OUTPUT_TOKENS = 461.97
V2_MEAN_LATENCY_SECONDS = 4.246715


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def stable_shard(job_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(str(job_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def shard_name(index: int, count: int) -> str:
    width = max(3, len(str(count)))
    return f"p1_{index + 1:0{width}d}_of_{count:0{width}d}.csv"


def run_id(prefix: str, index: int, count: int) -> str:
    width = max(3, len(str(count)))
    return f"{prefix}-shard-{index + 1:0{width}d}-of-{count:0{width}d}"


def tier2_parallel_plan(workers: int, shard_count: int) -> dict[str, float | int]:
    """Project synchronous load from the completed v2 pilot.

    This is a launch guard, not a distributed rate limiter. The one-request-at-
    a-time shard workers make concurrency a useful conservative control, while
    the startup interval avoids an instantaneous 32-request burst.
    """
    if workers <= 0:
        raise ValueError("--max-parallel must be positive")
    if shard_count <= 0:
        raise ValueError("Prepared shard count must be positive")

    total_tokens = V2_MEAN_INPUT_TOKENS + V2_MEAN_OUTPUT_TOKENS
    per_worker_rpm = 60.0 / V2_MEAN_LATENCY_SECONDS
    rpm_safe_cap = int(
        TIER2_REQUESTS_PER_MINUTE * TIER2_TARGET_UTILIZATION / per_worker_rpm
    )
    tpm_safe_cap = int(
        TIER2_TOKENS_PER_MINUTE
        * TIER2_TARGET_UTILIZATION
        / (per_worker_rpm * total_tokens)
    )
    model_safe_cap = max(1, min(rpm_safe_cap, tpm_safe_cap))
    recommended_cap = min(shard_count, model_safe_cap)
    projected_rpm = workers * per_worker_rpm
    projected_input_tpm = projected_rpm * V2_MEAN_INPUT_TOKENS
    projected_total_tpm = projected_rpm * total_tokens
    projected_minutes = DEFAULT_EXPECTED_ROWS * V2_MEAN_LATENCY_SECONDS / (60.0 * workers)
    return {
        "workers": workers,
        "shard_count": shard_count,
        "rpm_safe_cap": rpm_safe_cap,
        "tpm_safe_cap": tpm_safe_cap,
        "model_safe_cap": model_safe_cap,
        "recommended_cap": recommended_cap,
        "projected_rpm": projected_rpm,
        "projected_input_tpm": projected_input_tpm,
        "projected_total_tpm": projected_total_tpm,
        "projected_minutes": projected_minutes,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config, require_keys=False)
    models = config["models"]
    if len(models) != 1:
        raise ValueError("P1 production config must contain exactly one model")
    model = models[0]
    expected = {
        "provider": "openai",
        "model_id": "gpt-5.6-luna",
        "reasoning_effort": "medium",
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"P1 production requires {key}={value!r}; found {model.get(key)!r}")
    if config.get("max_cost_usd") is None:
        raise ValueError("P1 production config must define max_cost_usd for every shard")
    return config


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(canonical)


def load_preparation(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Preparation manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("preparation_version") != PREPARATION_VERSION:
        raise ValueError("Unsupported P1 preparation manifest version")
    return manifest


def validate_prepared_files(directory: Path, manifest: dict[str, Any]) -> None:
    master = directory / manifest["master_input"]["file"]
    if not master.exists() or sha256_file(master) != manifest["master_input"]["sha256"]:
        raise ValueError(f"Prepared master input is missing or changed: {master}")
    total = 0
    seen_files: set[str] = set()
    for shard in manifest["shards"]:
        name = shard["file"]
        if name in seen_files:
            raise ValueError(f"Duplicate shard filename in manifest: {name}")
        seen_files.add(name)
        path = directory / name
        if not path.exists() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Prepared shard is missing or changed: {path}")
        total += int(shard["rows"])
    if total != int(manifest["rows"]):
        raise ValueError(f"Shard row total {total} does not match manifest rows {manifest['rows']}")


def prepare(args: argparse.Namespace) -> None:
    source = project_path(args.source)
    output = project_path(args.output_dir)
    if not source.exists():
        raise FileNotFoundError(f"P1 source does not exist: {source}")
    if args.shards <= 0:
        raise ValueError("--shards must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty preparation directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source, columns=SOURCE_COLUMNS)
    frame = frame.loc[frame["filter_priority"].eq(P1), SOURCE_COLUMNS[:-1]].copy()
    frame["job_id"] = frame["job_id"].astype(str).str.strip()
    if frame["job_id"].eq("").any() or frame["job_id"].duplicated().any():
        raise ValueError("P1 source must contain unique, non-empty job_id values")
    if args.expected_rows and len(frame) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows:,} P1 rows but found {len(frame):,}; "
            "inspect the upstream cohort or pass the intentional count explicitly"
        )

    for column in ("title_raw", "jobtitle_translated", "description"):
        frame[column] = frame[column].fillna("").astype(str)
    numeric_order = pd.to_numeric(frame["job_id"], errors="coerce")
    frame = frame.assign(_numeric_order=numeric_order).sort_values(
        ["_numeric_order", "job_id"], kind="stable", na_position="last"
    ).drop(columns="_numeric_order").reset_index(drop=True)
    frame["pilot_stratum"] = P1
    frame = frame[INPUT_COLUMNS]

    master_path = output / "p1_all.csv"
    write_csv_atomic(master_path, frame)
    shard_indexes = frame["job_id"].map(lambda value: stable_shard(value, args.shards))
    shards: list[dict[str, Any]] = []
    assigned_ids: set[str] = set()
    for index in range(args.shards):
        part = frame.loc[shard_indexes.eq(index)].copy()
        path = output / shard_name(index, args.shards)
        write_csv_atomic(path, part)
        ids = set(part["job_id"])
        if assigned_ids & ids:
            raise AssertionError("A job_id was assigned to more than one shard")
        assigned_ids.update(ids)
        shards.append(
            {
                "index": index,
                "number": index + 1,
                "file": path.name,
                "rows": len(part),
                "sha256": sha256_file(path),
            }
        )
    if assigned_ids != set(frame["job_id"]):
        raise AssertionError("Prepared shards do not cover the complete P1 cohort")

    selection = "\n".join(frame["job_id"].tolist()).encode("utf-8")
    manifest = {
        "preparation_version": PREPARATION_VERSION,
        "cohort": P1,
        "rows": len(frame),
        "unique_job_ids": int(frame["job_id"].nunique()),
        "shard_count": args.shards,
        "shard_rule": "sha256(job_id) first 8 bytes modulo shard_count",
        "selection_sha256": sha256_bytes(selection),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "input_columns": INPUT_COLUMNS,
        "model_input_excludes_filter_priority": True,
        "model_input_excludes_human_labels": True,
        "runner_prompt_excludes_pilot_stratum": True,
        "master_input": {
            "file": master_path.name,
            "rows": len(frame),
            "sha256": sha256_file(master_path),
        },
        "shards": shards,
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(f"Prepared {len(frame):,} P1 postings in {args.shards} deterministic shards")
    print(f"Manifest: {output / 'manifest.json'}")


def command_for_shard(
    *,
    input_path: Path,
    config_path: Path,
    prompt_sha256: str,
    shard_run_id: str,
    retry_ambiguous: bool,
    accept_ambiguous: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "revelio_pilot.run_pilot",
        "--input",
        str(input_path),
        "--config",
        str(config_path),
        "--phase",
        "frozen_test",
        "--prompt-version",
        P1_PROMPT_VERSION,
        "--run-id",
        shard_run_id,
        "--require-prompt-hash",
        prompt_sha256,
    ]
    if retry_ambiguous:
        command.append("--retry-ambiguous-in-flight")
    if accept_ambiguous:
        command.append("--accept-ambiguous-in-flight-as-failure")
    return command


def launch(args: argparse.Namespace) -> None:
    prepared = project_path(args.prepared_dir)
    manifest = load_preparation(prepared)
    validate_prepared_files(prepared, manifest)
    config_path = project_path(args.config)
    config = load_config(config_path)
    shard_count = int(manifest["shard_count"])
    plan = tier2_parallel_plan(args.max_parallel, shard_count)
    if args.max_parallel > shard_count:
        raise ValueError(
            f"--max-parallel cannot exceed the {shard_count} prepared shards"
        )
    if args.max_parallel > plan["model_safe_cap"]:
        raise ValueError(
            f"--max-parallel={args.max_parallel} exceeds the v2-based Tier 2 safety cap "
            f"of {plan['model_safe_cap']} workers"
        )
    if args.start_interval_seconds < 0:
        raise ValueError("--start-interval-seconds must be non-negative")
    current_hash = P1_PROMPT_SHA256
    if args.prompt_hash != current_hash:
        raise ValueError(f"Prompt hash mismatch: current={current_hash}")

    aggregate_cap = float(config["max_cost_usd"]) * int(manifest["shard_count"])
    print(
        f"Launching {manifest['rows']:,} P1 postings across {manifest['shard_count']} shards; "
        f"at most {args.max_parallel} processes; aggregate configured ceiling USD {aggregate_cap:.2f}"
    )
    print(
        "Tier 2 v2-pilot projection: "
        f"{plan['projected_rpm']:.0f}/{TIER2_REQUESTS_PER_MINUTE:,} RPM, "
        f"{plan['projected_total_tpm']:,.0f}/{TIER2_TOKENS_PER_MINUTE:,} observed tokens/min, "
        f"about {plan['projected_minutes']:.0f} minutes before overhead; "
        f"safe model cap={plan['model_safe_cap']}, prepared-shard cap={shard_count}"
    )
    queue: deque[tuple[dict[str, Any], list[str]]] = deque()
    for shard in manifest["shards"]:
        shard_run_id = run_id(args.run_prefix, shard["index"], manifest["shard_count"])
        command = command_for_shard(
            input_path=(prepared / shard["file"]).resolve(),
            config_path=config_path,
            prompt_sha256=args.prompt_hash,
            shard_run_id=shard_run_id,
            retry_ambiguous=args.retry_ambiguous_in_flight,
            accept_ambiguous=args.accept_ambiguous_in_flight_as_failure,
        )
        queue.append((shard, command))

    if args.dry_run:
        for shard, command in queue:
            print(f"[{shard['number']}/{manifest['shard_count']}] {subprocess.list2cmdline(command)}")
        print("DRY RUN: no child processes or API calls were started")
        return

    log_dir = PROJECT_ROOT / "outputs" / "p1_launcher_logs" / args.run_prefix
    log_dir.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[subprocess.Popen, Any, Any, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    try:
        while queue or active:
            while queue and len(active) < args.max_parallel:
                shard, command = queue.popleft()
                stem = Path(shard["file"]).stem
                stdout_handle = (log_dir / f"{stem}.stdout.log").open("a", encoding="utf-8")
                stderr_handle = (log_dir / f"{stem}.stderr.log").open("a", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
                active[process.pid] = (process, stdout_handle, stderr_handle, shard)
                print(f"Started shard {shard['number']}/{manifest['shard_count']} (pid={process.pid})")
                if queue and len(active) < args.max_parallel and args.start_interval_seconds:
                    time.sleep(args.start_interval_seconds)

            finished = []
            for pid, (process, stdout_handle, stderr_handle, shard) in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                stdout_handle.close()
                stderr_handle.close()
                finished.append(pid)
                print(
                    f"Finished shard {shard['number']}/{manifest['shard_count']} "
                    f"with exit code {return_code}"
                )
                if return_code:
                    failures.append({"shard": shard["number"], "exit_code": return_code})
            for pid in finished:
                del active[pid]
            if active and not finished:
                time.sleep(0.5)
    except BaseException:
        for process, stdout_handle, stderr_handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
            stdout_handle.close()
            stderr_handle.close()
        raise
    if failures:
        raise SystemExit(f"One or more shards exited unsuccessfully: {failures}")


def local_status(args: argparse.Namespace) -> None:
    prepared = project_path(args.prepared_dir)
    preparation = load_preparation(prepared)
    model_name = load_config(project_path(args.config))["models"][0]["name"]
    rows = []
    for shard in preparation["shards"]:
        shard_run_id = run_id(args.run_prefix, shard["index"], preparation["shard_count"])
        manifest_path = PROJECT_ROOT / "outputs" / "runs" / shard_run_id / "manifest.json"
        if manifest_path.exists():
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = run_manifest.get("run_status", "unknown")
            completed = int(run_manifest.get("completed_job_model_pairs", 0))
            succeeded = int(run_manifest.get("successful_calls", 0))
            failed = int(run_manifest.get("terminal_failures_recorded", 0))
            cost = float(run_manifest.get("estimated_cost_usd", 0) or 0)
        else:
            status, completed, succeeded, failed, cost = "not_started", 0, 0, 0, 0.0
        rows.append(
            {
                "shard": shard["number"],
                "rows": shard["rows"],
                "model": model_name,
                "status": status,
                "completed": completed,
                "succeeded": succeeded,
                "failed": failed,
                "estimated_cost_usd": cost,
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(
        f"TOTAL completed={table['completed'].sum():,}/{preparation['rows']:,} "
        f"success={table['succeeded'].sum():,} failures={table['failed'].sum():,} "
        f"estimated_cost_usd={table['estimated_cost_usd'].sum():.4f}"
    )


def prediction_row(source: dict[str, str], record: dict[str, Any]) -> dict[str, Any]:
    prediction = record["prediction"]
    return {
        "job_id": str(record["job_id"]),
        "title_raw": source.get("title_raw", ""),
        "jobtitle_translated": source.get("jobtitle_translated", ""),
        "pilot_stratum": record.get("pilot_stratum", ""),
        "model_name": record.get("model_name", ""),
        **{field: prediction.get(field, "") for field in REQUIRED_FIELDS},
        "validation_warnings": " | ".join(record.get("validation_warnings", [])),
        "input_tokens": record.get("input_tokens", 0),
        "output_tokens": record.get("output_tokens", 0),
        "latency_seconds": record.get("latency_seconds", 0),
        "estimated_cost_usd": record.get("estimated_cost_usd", 0),
        "attempts": record.get("attempts", 0),
    }


def consolidate(args: argparse.Namespace) -> None:
    prepared = project_path(args.prepared_dir)
    preparation = load_preparation(prepared)
    validate_prepared_files(prepared, preparation)
    config_path = project_path(args.config)
    config = load_config(config_path)
    expected_config_hash = config_sha256(config)
    model_name = config["models"][0]["name"]
    master_path = prepared / preparation["master_input"]["file"]
    master = pd.read_csv(master_path, dtype={"job_id": str}).fillna("")
    source_by_id = {str(row["job_id"]): row.to_dict() for _, row in master.iterrows()}

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    for shard in preparation["shards"]:
        shard_run_id = run_id(args.run_prefix, shard["index"], preparation["shard_count"])
        directory = PROJECT_ROOT / "outputs" / "runs" / shard_run_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run manifest is missing for shard {shard['number']}: {manifest_path}")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("run_status") != "completed":
            raise ValueError(
                f"Shard {shard['number']} is not complete: {run_manifest.get('run_status')}"
            )
        if run_manifest.get("prompt_version") != P1_PROMPT_VERSION:
            raise ValueError(f"Shard {shard['number']} used a different prompt version")
        if run_manifest.get("prompt_sha256") != P1_PROMPT_SHA256:
            raise ValueError(f"Shard {shard['number']} used a different prompt/schema bundle")
        if run_manifest.get("models") != config["models"]:
            raise ValueError(f"Shard {shard['number']} used a different model config")
        if run_manifest.get("config_sha256") != expected_config_hash:
            raise ValueError(f"Shard {shard['number']} used a different run configuration")
        if run_manifest.get("input_sha256") != shard["sha256"]:
            raise ValueError(f"Shard {shard['number']} used a different input file")
        if int(run_manifest.get("selected_rows", -1)) != int(shard["rows"]):
            raise ValueError(f"Shard {shard['number']} row count differs from preparation manifest")
        shard_manifests.append(run_manifest)
        predictions.extend(load_jsonl(directory / "predictions.jsonl"))
        failures.extend(load_jsonl(directory / "failures.jsonl"))

    prediction_keys = [(str(row["job_id"]), str(row["model_name"])) for row in predictions]
    failure_keys = [(str(row["job_id"]), str(row["model_name"])) for row in failures]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise ValueError("Duplicate job/model pairs found across shard predictions")
    if len(failure_keys) != len(set(failure_keys)):
        raise ValueError("Duplicate job/model pairs found across shard failures")
    if set(prediction_keys) & set(failure_keys):
        raise ValueError("A job/model pair appears in both predictions and failures")
    expected_keys = {(job_id, model_name) for job_id in source_by_id}
    observed_keys = set(prediction_keys) | set(failure_keys)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)[:10]
        unexpected = sorted(observed_keys - expected_keys)[:10]
        raise ValueError(f"Consolidated coverage mismatch; missing={missing}, unexpected={unexpected}")

    order = {job_id: index for index, job_id in enumerate(master["job_id"].astype(str))}
    predictions.sort(key=lambda row: order[str(row["job_id"])])
    failures.sort(key=lambda row: order[str(row["job_id"])])
    output = project_path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output / "predictions.jsonl", predictions)
    write_jsonl_atomic(output / "failures.jsonl", failures)

    flat = pd.DataFrame([prediction_row(source_by_id[str(row["job_id"])], row) for row in predictions])
    write_csv_atomic(output / "p1_results.csv", flat)
    failure_columns = [
        "job_id", "pilot_stratum", "model_name", "provider", "model_id",
        "error_type", "error", "attempts", "input_tokens", "output_tokens",
        "latency_seconds", "estimated_cost_usd", "cost_unknown",
        "ambiguous_in_flight_attempts", "failed_at_utc",
    ]
    failure_frame = pd.DataFrame(failures, columns=failure_columns)
    write_csv_atomic(output / "p1_failures.csv", failure_frame)

    summary_groups = Counter(row["prediction"]["summary_group"] for row in predictions)
    seo_duties = Counter(row["prediction"]["seo_duty"] for row in predictions)
    geo_duties = Counter(row["prediction"]["geo_duty"] for row in predictions)
    summary = {
        "consolidation_version": CONSOLIDATION_VERSION,
        "cohort": P1,
        "model": config["models"][0],
        "rows": preparation["rows"],
        "successful_predictions": len(predictions),
        "terminal_failures": len(failures),
        "complete_coverage": len(predictions) + len(failures) == preparation["rows"],
        "summary_group_counts": dict(sorted(summary_groups.items())),
        "seo_duty_counts": dict(sorted(seo_duties.items())),
        "geo_duty_counts": dict(sorted(geo_duties.items())),
        "input_tokens": sum(int(row.get("input_tokens", 0) or 0) for row in predictions + failures),
        "output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in predictions + failures),
        "estimated_cost_usd": sum(
            float(row.get("estimated_cost_usd", 0) or 0) for row in predictions + failures
        ),
        "prompt_version": P1_PROMPT_VERSION,
        "prompt_sha256": P1_PROMPT_SHA256,
        "schema_sha256": sha256_bytes(
            json.dumps(PREDICTION_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "preparation_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "source_master_sha256": preparation["master_input"]["sha256"],
        "shard_run_ids": [manifest["run_id"] for manifest in shard_manifests],
        "note": "Production P1 classifications are model outputs, not human gold labels.",
    }
    write_json_atomic(output / "summary.json", summary)
    print(
        f"Consolidated {len(predictions):,} predictions and {len(failures):,} terminal failures "
        f"into {output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create the blind full-P1 input and shards")
    prepare_parser.add_argument("--source", default=DEFAULT_SOURCE)
    prepare_parser.add_argument("--output-dir", default=DEFAULT_PREPARED_DIR)
    prepare_parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    prepare_parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    prepare_parser.set_defaults(fn=prepare)

    launch_parser = subparsers.add_parser("launch", help="Run prepared shards through the pilot runner")
    launch_parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    launch_parser.add_argument("--config", default=DEFAULT_CONFIG)
    launch_parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    launch_parser.add_argument("--prompt-hash", required=True)
    launch_parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    launch_parser.add_argument(
        "--start-interval-seconds",
        type=float,
        default=DEFAULT_START_INTERVAL_SECONDS,
        help="Delay between shard-process starts to soften the initial API traffic ramp",
    )
    launch_parser.add_argument("--dry-run", action="store_true")
    ambiguous = launch_parser.add_mutually_exclusive_group()
    ambiguous.add_argument("--retry-ambiguous-in-flight", action="store_true")
    ambiguous.add_argument("--accept-ambiguous-in-flight-as-failure", action="store_true")
    launch_parser.set_defaults(fn=launch)

    status_parser = subparsers.add_parser("status", help="Show local progress for every shard")
    status_parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    status_parser.add_argument("--config", default=DEFAULT_CONFIG)
    status_parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    status_parser.set_defaults(fn=local_status)

    consolidate_parser = subparsers.add_parser(
        "consolidate", help="Validate complete shard coverage and create P1 result files"
    )
    consolidate_parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    consolidate_parser.add_argument("--config", default=DEFAULT_CONFIG)
    consolidate_parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    consolidate_parser.add_argument("--output-dir", default="outputs/p1_luna_medium")
    consolidate_parser.add_argument("--overwrite", action="store_true")
    consolidate_parser.set_defaults(fn=consolidate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
