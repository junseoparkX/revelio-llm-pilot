from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .prompt import get_system_prompt, prompt_hash
from .run_pilot import INPUT_COLUMNS, load_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_jobs_hash(frame: pd.DataFrame) -> str:
    payload = json.dumps(
        frame[INPUT_COLUMNS].to_dict(orient="records"),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def index_records(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        job_id = str(record["job_id"])
        if job_id in indexed:
            raise ValueError(f"Duplicate {label} record for job {job_id}")
        indexed[job_id] = record
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-input", required=True)
    parser.add_argument("--shard-run-glob", required=True)
    parser.add_argument("--recovery-run")
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt-version", required=True)
    args = parser.parse_args()

    master_path = Path(args.master_input)
    master = pd.read_csv(master_path, dtype={"job_id": str}).fillna("")
    expected_ids = master["job_id"].astype(str).tolist()
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Master input contains duplicate job_id values")

    run_root = Path("outputs/runs")
    shard_dirs = sorted(path for path in run_root.glob(args.shard_run_glob) if path.is_dir())
    if not shard_dirs:
        raise ValueError("No shard runs matched")

    manifests = []
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for directory in shard_dirs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("run_status") != "completed":
            raise ValueError(f"Shard is not complete: {directory}")
        if manifest.get("prompt_sha256") != prompt_hash(get_system_prompt(args.prompt_version)):
            raise ValueError(f"Prompt mismatch in {directory}")
        if [model.get("name") for model in manifest.get("models", [])] != [args.model_name]:
            raise ValueError(f"Model mismatch in {directory}")
        manifests.append(manifest)
        predictions.extend(load_jsonl(directory / "predictions.jsonl"))
        failure_path = directory / "failures.jsonl"
        if failure_path.exists():
            failures.extend(load_jsonl(failure_path))
        for record in load_jsonl(directory / "raw_responses.jsonl"):
            raw_records.append({**record, "source_run_id": manifest["run_id"]})

    prediction_index = index_records(predictions, "shard prediction")
    failure_index = index_records(failures, "shard failure")
    overlap = set(prediction_index) & set(failure_index)
    if overlap:
        raise ValueError(f"Jobs occur in both shard predictions and failures: {sorted(overlap)}")
    if set(prediction_index) | set(failure_index) != set(expected_ids):
        raise ValueError("Shard outputs do not cover the master input exactly once")

    recovered_ids: list[str] = []
    recovery_manifest = None
    if args.recovery_run:
        recovery_dir = run_root / args.recovery_run
        recovery_manifest = json.loads((recovery_dir / "manifest.json").read_text(encoding="utf-8"))
        if recovery_manifest.get("run_status") != "completed":
            raise ValueError("Recovery run is not complete")
        retry_predictions = index_records(
            load_jsonl(recovery_dir / "predictions.jsonl"), "recovery prediction"
        )
        retry_failures = load_jsonl(recovery_dir / "failures.jsonl")
        if retry_failures:
            raise ValueError("Recovery run still contains failures")
        if set(retry_predictions) != set(failure_index):
            raise ValueError("Recovery predictions must match the first-pass failures exactly")
        for record in load_jsonl(recovery_dir / "raw_responses.jsonl"):
            retry_raw = {**record, "attempt": int(record.get("attempt", 1)) + 1}
            raw_records.append({**retry_raw, "source_run_id": recovery_manifest["run_id"]})
        for job_id, retry in retry_predictions.items():
            first = failure_index.pop(job_id)
            merged = dict(retry)
            for key in ("input_tokens", "output_tokens"):
                merged[key] = int(first.get(key, 0)) + int(retry.get(key, 0))
            for key in ("latency_seconds", "estimated_cost_usd"):
                merged[key] = float(first.get(key, 0) or 0) + float(retry.get(key, 0) or 0)
            merged["attempts"] = int(first.get("attempts", 1)) + int(retry.get("attempts", 1))
            merged["first_attempt_status"] = "FAILURE"
            merged["first_attempt_error_type"] = first.get("error_type", "")
            merged["recovered_with_max_output_tokens"] = recovery_manifest.get(
                "max_output_tokens"
            )
            prediction_index[job_id] = merged
            recovered_ids.append(job_id)

    if failure_index:
        raise ValueError(f"Unrecovered failures remain: {sorted(failure_index)}")

    ordered_predictions = []
    for job_id in expected_ids:
        record = dict(prediction_index[job_id])
        record.setdefault("first_attempt_status", "SUCCESS")
        ordered_predictions.append(record)

    output = run_root / args.output_run_id
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "predictions.jsonl", ordered_predictions)
    write_jsonl(output / "raw_responses.jsonl", raw_records)
    write_jsonl(output / "failures.jsonl", [])

    usage_fields = [
        "job_id", "pilot_stratum", "model_name", "provider", "model_id",
        "input_tokens", "output_tokens", "latency_seconds", "estimated_cost_usd", "attempts",
    ]
    with (output / "usage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=usage_fields)
        writer.writeheader()
        for record in ordered_predictions:
            writer.writerow({key: record.get(key, "") for key in usage_fields})

    prompts = output / "prompts"
    prompts.mkdir()
    shutil.copyfile(shard_dirs[0] / "prompts" / "prompt.txt", prompts / "prompt.txt")
    shutil.copyfile(shard_dirs[0] / "prompts" / "schema.json", prompts / "schema.json")

    first = manifests[0]
    total_cost = sum(float(record.get("estimated_cost_usd", 0) or 0) for record in raw_records)
    manifest = {
        "run_id": args.output_run_id,
        "phase": "frozen_test",
        "input": str(master_path),
        "input_sha256": sha256_file(master_path),
        "selected_jobs_sha256": selected_jobs_hash(master),
        "selected_rows": len(master),
        "prompt_version": args.prompt_version,
        "prompt_sha256": first["prompt_sha256"],
        "schema_sha256": first["schema_sha256"],
        "config_sha256": first["config_sha256"],
        "models": first["models"],
        "seed": first.get("seed"),
        "max_cost_usd": sum(float(item.get("max_cost_usd", 0) or 0) for item in manifests),
        "max_retries": first.get("max_retries", 0),
        "timeout_seconds": first.get("timeout_seconds"),
        "max_output_tokens": first.get("max_output_tokens"),
        "package_version": first.get("package_version"),
        "expected_calls": len(master),
        "successful_calls": len(ordered_predictions),
        "terminal_failures_recorded": 0,
        "completed_job_model_pairs": len(ordered_predictions),
        "estimated_cost_usd": total_cost,
        "stopped_for_budget": False,
        "run_status": "completed",
        "started_at_utc": min(item["started_at_utc"] for item in manifests),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_pass_successful_calls": len(master) - len(recovered_ids),
        "first_pass_failures": len(recovered_ids),
        "recovered_calls": len(recovered_ids),
        "recovery_max_output_tokens": (
            recovery_manifest.get("max_output_tokens") if recovery_manifest else None
        ),
        "recovered_job_ids": sorted(recovered_ids),
        "shard_run_ids": [item["run_id"] for item in manifests],
        "recovery_run_id": recovery_manifest.get("run_id") if recovery_manifest else None,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Consolidated {len(ordered_predictions)} predictions; "
        f"recovered {len(recovered_ids)}; recorded cost USD {total_cost:.6f}"
    )


if __name__ == "__main__":
    main()
