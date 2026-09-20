from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from . import __version__
from .prompt import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    build_user_prompt,
    get_system_prompt,
    prompt_hash,
)
from .providers import call, required_key
from .schema import PREDICTION_SCHEMA, normalize_prediction, validate_prediction


INPUT_COLUMNS = ["job_id", "title_raw", "jobtitle_translated", "description", "pilot_stratum"]
CHECKPOINT_VERSION = 1


class RunLock:
    """Cross-process lock that is automatically released when a process exits."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RuntimeError(
                f"Run directory is already active in another process: {self.path.parent}"
            ) from exc
        metadata = json.dumps({"pid": os.getpid(), "acquired_at_utc": now_utc()}).encode("utf-8")
        handle.seek(1)
        handle.truncate()
        handle.write(metadata)
        handle.flush()
        os.fsync(handle.fileno())
        self.handle = handle

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def replace_with_retry(temporary: Path, target: Path) -> None:
    for attempt in range(6):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Windows can briefly deny an atomic replace while an editor or
            # progress reader has the destination open.
            time.sleep(0.05 * (2**attempt))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def record_key(record: dict) -> tuple[str, str]:
    return str(record["job_id"]), str(record["model_name"])


def raw_record_key(record: dict) -> tuple[str, str, int]:
    return str(record["job_id"]), str(record["model_name"]), int(record["attempt"])


def checkpoint_path(directory: Path, job_id: str, model_name: str) -> Path:
    digest = hashlib.sha256(
        json.dumps([str(job_id), str(model_name)], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return directory / f"{digest}.json"


def save_checkpoint(directory: Path, state: dict) -> None:
    value = dict(state)
    value["checkpoint_version"] = CHECKPOINT_VERSION
    value["updated_at_utc"] = now_utc()
    write_json(
        checkpoint_path(directory, str(value["job_id"]), str(value["model_name"])),
        value,
    )


def load_checkpoints(directory: Path) -> dict[tuple[str, str], dict]:
    states: dict[tuple[str, str], dict] = {}
    if not directory.exists():
        return states
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version in {path}")
        key = record_key(value)
        if checkpoint_path(directory, *key) != path:
            raise ValueError(f"Checkpoint filename does not match its job/model identity: {path}")
        if key in states:
            raise ValueError(f"Duplicate checkpoint for job/model {key}")
        states[key] = value
    return states


def unique_records_by_key(records: list[dict], label: str) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for record in records:
        key = record_key(record)
        if key in indexed:
            raise ValueError(f"Duplicate {label} record for job/model {key}")
        indexed[key] = record
    return indexed


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path} line {line_number}")
        records.append(value)
    return records


def repair_jsonl_tail(path: Path) -> str | None:
    """Repair only a non-newline-terminated final JSONL record.

    Interior corruption is intentionally left untouched so load_jsonl reports it.
    Every call record is checkpointed before its JSONL append, so an incomplete
    final record can be recreated during startup recovery.
    """
    if not path.exists():
        return None
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    line_start = data.rfind(b"\n") + 1
    tail = data[line_start:]
    try:
        value = json.loads(tail.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSONL tail is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        with path.open("r+b") as handle:
            handle.truncate(line_start)
            handle.flush()
            os.fsync(handle.fileno())
        return "removed_incomplete_final_record"
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return "added_missing_final_newline"


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    fence = "\x60\x60\x60"
    if cleaned.startswith(fence):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


def usage_tokens(provider: str, usage: dict) -> tuple[int, int]:
    if provider == "openai":
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    if provider == "mistral":
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    if provider == "gemini":
        return int(usage.get("promptTokenCount", 0)), (
            int(usage.get("candidatesTokenCount", 0))
            + int(usage.get("thoughtsTokenCount", 0))
        )
    return 0, 0


def validate_config(config: dict, require_keys: bool) -> None:
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("Config must define a non-empty models list")
    names = []
    for model in config["models"]:
        for field in ("name", "provider", "model_id", "input_usd_per_million", "output_usd_per_million"):
            if field not in model:
                raise ValueError(f"Model config is missing {field}")
        if "REPLACE_WITH" in str(model["model_id"]):
            raise ValueError(f"Model {model['name']} still has a placeholder model_id")
        required_key(model["provider"])
        reasoning_effort = model.get("reasoning_effort")
        if reasoning_effort is not None:
            if model["provider"] != "openai":
                raise ValueError("reasoning_effort is currently supported only for OpenAI models")
            if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
                raise ValueError(
                    f"Model {model['name']} has an invalid reasoning_effort"
                )
        for price_field in ("input_usd_per_million", "output_usd_per_million"):
            price = model[price_field]
            if not isinstance(price, (int, float)) or isinstance(price, bool) or price < 0:
                raise ValueError(f"Model {model['name']} has an invalid {price_field}")
        names.append(model["name"])
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique")
    max_output_tokens = config.get("max_output_tokens")
    if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        raise ValueError("Config must define a positive integer max_output_tokens")
    max_retries = config.get("max_retries", 0)
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    timeout_seconds = config.get("timeout_seconds", 90)
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    max_cost = config.get("max_cost_usd")
    if max_cost is not None and (
        not isinstance(max_cost, (int, float)) or isinstance(max_cost, bool) or max_cost <= 0
    ):
        raise ValueError("max_cost_usd must be positive when supplied")
    if require_keys:
        missing = sorted(
            {
                required_key(model["provider"])
                for model in config["models"]
                if not os.environ.get(required_key(model["provider"]), "").strip()
            }
        )
        if missing:
            raise RuntimeError(f"Missing API credentials: {', '.join(missing)}. No API calls were made.")


def validate_input(frame: pd.DataFrame) -> None:
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if frame["job_id"].isna().any() or frame["job_id"].astype(str).str.strip().eq("").any():
        raise ValueError("job_id must be non-empty")
    if frame["job_id"].astype(str).duplicated().any():
        raise ValueError("job_id must be unique")


def classify_error(error: Exception) -> str:
    if isinstance(error, ValueError) and "no output text" in str(error):
        return "empty_output"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, ValueError) and (
        "fields" in str(error) or "must be" in str(error) or "evidence" in str(error)
    ):
        return "schema_or_semantic_validation"
    return "provider_or_runtime"


def rebuild_usage(prediction_path: Path, usage_path: Path) -> None:
    fields = [
        "job_id",
        "pilot_stratum",
        "model_name",
        "provider",
        "model_id",
        "input_tokens",
        "output_tokens",
        "latency_seconds",
        "estimated_cost_usd",
        "attempts",
    ]
    records = load_jsonl(prediction_path)
    temporary = usage_path.with_suffix(usage_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, usage_path)


def update_manifest_progress(
    manifest_path: Path,
    manifest: dict,
    prediction_count: int,
    failure_count: int,
    total_cost: float,
    stopped_for_budget: bool,
    run_status: str,
) -> None:
    manifest.update(
        {
            "successful_calls": prediction_count,
            "terminal_failures_recorded": failure_count,
            "completed_job_model_pairs": prediction_count + failure_count,
            "estimated_cost_usd": total_cost,
            "stopped_for_budget": stopped_for_budget,
            "run_status": run_status,
            "last_checkpoint_at_utc": now_utc(),
        }
    )
    write_json(manifest_path, manifest)


def commit_terminal_checkpoint(
    checkpoint_directory: Path,
    output_path: Path,
    indexed_records: dict[tuple[str, str], dict],
    state: dict,
    status: str,
    record: dict,
) -> None:
    key = record_key(record)
    terminal_state = dict(state)
    terminal_state.update({"status": status, "terminal_record": record})
    # The atomic checkpoint is written first. If the process stops before the
    # JSONL append, startup recovery can recreate the missing terminal record.
    save_checkpoint(checkpoint_directory, terminal_state)
    if key not in indexed_records:
        append_jsonl(output_path, record)
        indexed_records[key] = record


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["instruction_check", "frozen_test"], required=True)
    parser.add_argument(
        "--prompt-version",
        choices=sorted(PROMPT_VERSIONS),
        default=DEFAULT_PROMPT_VERSION,
        help="Versioned classification prompt to freeze for this run.",
    )
    parser.add_argument("--require-prompt-hash")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    ambiguous_group = parser.add_mutually_exclusive_group()
    ambiguous_group.add_argument(
        "--retry-ambiguous-in-flight",
        action="store_true",
        help=(
            "Retry a request that was in flight when an earlier process stopped. "
            "This can duplicate a billable provider call, so it is never automatic."
        ),
    )
    ambiguous_group.add_argument(
        "--accept-ambiguous-in-flight-as-failure",
        action="store_true",
        help=(
            "Record a request interrupted in flight as a terminal failure without retrying it."
        ),
    )
    args = parser.parse_args()

    selected_system_prompt = get_system_prompt(args.prompt_version)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config, require_keys=not args.dry_run)
    current_prompt_hash = prompt_hash(selected_system_prompt)
    schema_hash = hashlib.sha256(
        json.dumps(PREDICTION_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.phase == "frozen_test" and not args.require_prompt_hash:
        raise SystemExit("--require-prompt-hash is mandatory for frozen_test")
    if args.require_prompt_hash and args.require_prompt_hash != current_prompt_hash:
        raise SystemExit(f"Prompt hash mismatch: current={current_prompt_hash}")

    input_path = Path(args.input)
    frame = pd.read_csv(input_path, dtype={"job_id": str}).fillna("")
    validate_input(frame)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        frame = frame.head(args.limit)
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    selected_jobs_hash = hashlib.sha256(
        json.dumps(frame[INPUT_COLUMNS].to_dict(orient="records"), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    expected_calls = len(frame) * len(config["models"])

    if args.dry_run:
        print(f"DRY RUN: {len(frame)} postings x {len(config['models'])} models = {expected_calls} calls")
        print(f"Prompt version: {args.prompt_version}")
        print(f"Prompt hash: {current_prompt_hash}")
        print(f"Schema hash: {schema_hash}")
        print("No API calls or output files were created.")
        return

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = Path("outputs/runs") / run_id
    manifest_path = output / "manifest.json"
    prediction_path = output / "predictions.jsonl"
    failure_path = output / "failures.jsonl"
    raw_path = output / "raw_responses.jsonl"
    usage_path = output / "usage.csv"
    checkpoint_directory = output / "checkpoints"
    run_lock = RunLock(output / ".run.lock")
    run_lock.acquire()

    manifest = {
        "run_id": run_id,
        "phase": args.phase,
        "input": str(input_path),
        "input_sha256": input_hash,
        "selected_jobs_sha256": selected_jobs_hash,
        "selected_rows": len(frame),
        "prompt_version": args.prompt_version,
        "prompt_sha256": current_prompt_hash,
        "schema_sha256": schema_hash,
        "config_sha256": config_hash,
        "models": config["models"],
        "seed": config.get("seed"),
        "max_cost_usd": config.get("max_cost_usd"),
        "max_retries": config.get("max_retries", 0),
        "timeout_seconds": config.get("timeout_seconds", 90),
        "max_output_tokens": config["max_output_tokens"],
        "package_version": __version__,
        "started_at_utc": now_utc(),
        "expected_calls": expected_calls,
    }

    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Manifests created before prompt versioning always used baseline_v1.
        if "prompt_version" not in existing_manifest:
            existing_manifest["prompt_version"] = DEFAULT_PROMPT_VERSION
        for key in (
            "phase", "input_sha256", "selected_jobs_sha256", "selected_rows", "prompt_version", "prompt_sha256",
            "schema_sha256", "config_sha256", "models", "seed", "max_cost_usd", "max_retries",
            "timeout_seconds", "max_output_tokens", "package_version", "expected_calls",
        ):
            if existing_manifest.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume run {run_id}: manifest mismatch for {key}")
        manifest = existing_manifest
    else:
        prompts_directory = output / "prompts"
        if output.exists():
            call_artifacts = [
                path for path in (prediction_path, failure_path, raw_path, usage_path)
                if path.exists()
            ]
            saved_checkpoints = list(checkpoint_directory.glob("*.json"))
            if call_artifacts or saved_checkpoints:
                raise ValueError(
                    f"Run {run_id} has call artifacts but no manifest; refusing an unsafe resume"
                )
        prompts_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        write_text(prompts_directory / "prompt.txt", selected_system_prompt)
        write_json(prompts_directory / "schema.json", PREDICTION_SCHEMA)
        write_json(manifest_path, manifest)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    stored_prompt_path = output / "prompts" / "prompt.txt"
    stored_schema_path = output / "prompts" / "schema.json"
    if not stored_prompt_path.exists() or stored_prompt_path.read_text(encoding="utf-8") != selected_system_prompt:
        raise ValueError(f"Stored prompt snapshot is missing or changed for run {run_id}")
    if not stored_schema_path.exists() or json.loads(
        stored_schema_path.read_text(encoding="utf-8")
    ) != PREDICTION_SCHEMA:
        raise ValueError(f"Stored schema snapshot is missing or changed for run {run_id}")

    recovery_events = []
    for jsonl_path in (prediction_path, failure_path, raw_path):
        action = repair_jsonl_tail(jsonl_path)
        if action:
            recovery_events.append(
                {"at_utc": now_utc(), "file": jsonl_path.name, "action": action}
            )
    if recovery_events:
        manifest.setdefault("recovery_events", []).extend(recovery_events)
        write_json(manifest_path, manifest)

    predictions_by_key = unique_records_by_key(load_jsonl(prediction_path), "prediction")
    failures_by_key = unique_records_by_key(load_jsonl(failure_path), "failure")
    overlap = set(predictions_by_key) & set(failures_by_key)
    if overlap:
        raise ValueError(f"Job/model pairs appear in both predictions and failures: {sorted(overlap)}")

    raw_records = load_jsonl(raw_path)
    raw_keys: set[tuple[str, str, int]] = set()
    for raw_record in raw_records:
        raw_key = raw_record_key(raw_record)
        if raw_key in raw_keys:
            raise ValueError(f"Duplicate raw response for job/model/attempt {raw_key}")
        raw_keys.add(raw_key)

    checkpoints = load_checkpoints(checkpoint_directory)
    valid_pairs = {
        (str(row["job_id"]), str(model["name"]))
        for _, row in frame.iterrows()
        for model in config["models"]
    }
    for key, state in checkpoints.items():
        if key not in valid_pairs:
            raise ValueError(f"Checkpoint does not belong to the selected input/config: {key}")
        status = state.get("status")
        if status not in {"in_flight", "response_received", "retry_pending", "succeeded", "failed"}:
            raise ValueError(f"Unknown checkpoint status for {key}: {status}")
        if status == "response_received":
            raw_record = state.get("raw_record")
            if not isinstance(raw_record, dict):
                raise ValueError(f"Response checkpoint is missing its raw record: {key}")
            raw_key = raw_record_key(raw_record)
            if raw_key not in raw_keys:
                append_jsonl(raw_path, raw_record)
                raw_records.append(raw_record)
                raw_keys.add(raw_key)
        if status in {"succeeded", "failed"}:
            record = state.get("terminal_record")
            if not isinstance(record, dict) or record_key(record) != key:
                raise ValueError(f"Terminal checkpoint is missing its matching record: {key}")
            target_index = predictions_by_key if status == "succeeded" else failures_by_key
            other_index = failures_by_key if status == "succeeded" else predictions_by_key
            target_path = prediction_path if status == "succeeded" else failure_path
            if key in other_index:
                raise ValueError(f"Terminal checkpoint conflicts with existing output for {key}")
            if key not in target_index:
                append_jsonl(target_path, record)
                target_index[key] = record

    done = set(predictions_by_key) | set(failures_by_key)
    total_cost = sum(float(record.get("estimated_cost_usd", 0) or 0) for record in raw_records)
    max_cost = config.get("max_cost_usd")
    stopped_for_budget = False
    update_manifest_progress(
        manifest_path, manifest, len(predictions_by_key), len(failures_by_key),
        total_cost, stopped_for_budget, "running",
    )

    try:
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            user_prompt = build_user_prompt(row_dict)
            posting_source = "\n".join(
                str(row_dict.get(column, ""))
                for column in ("title_raw", "jobtitle_translated", "description")
            )
            for model in config["models"]:
                key = (str(row_dict["job_id"]), model["name"])
                if key in done:
                    continue
                max_attempts = int(config.get("max_retries", 0)) + 1
                state = checkpoints.get(
                    key,
                    {
                        "job_id": key[0],
                        "model_name": key[1],
                        "provider": model["provider"],
                        "model_id": model["model_id"],
                        "status": "retry_pending",
                        "attempts": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latency_seconds": 0.0,
                        "estimated_cost_usd": 0.0,
                    },
                )

                while key not in done:
                    status = state["status"]
                    attempts = int(state.get("attempts", 0))
                    cumulative_input_tokens = int(state.get("input_tokens", 0))
                    cumulative_output_tokens = int(state.get("output_tokens", 0))
                    cumulative_latency = float(state.get("latency_seconds", 0))
                    cumulative_cost = float(state.get("estimated_cost_usd", 0))

                    if status == "in_flight":
                        if args.retry_ambiguous_in_flight:
                            state = {
                                **state,
                                "status": "retry_pending",
                                "attempts": max(attempts - 1, 0),
                                "ambiguous_in_flight_attempts": int(
                                    state.get("ambiguous_in_flight_attempts", 0)
                                ) + 1,
                                "cost_unknown": True,
                                "last_error_type": "interrupted_in_flight",
                                "last_error": (
                                    "An earlier in-flight request had no recorded response; "
                                    "the user explicitly authorized a replacement call."
                                ),
                            }
                            save_checkpoint(checkpoint_directory, state)
                            checkpoints[key] = state
                            continue
                        if not args.accept_ambiguous_in_flight_as_failure:
                            raise RuntimeError(
                                "An earlier request was interrupted in flight and may have been billed. "
                                "Restart with exactly one of --retry-ambiguous-in-flight or "
                                "--accept-ambiguous-in-flight-as-failure."
                            )
                        error_text = (
                            "A previous process stopped while this request was in flight. "
                            "It was not retried automatically because the provider may already have billed it."
                        )
                        record = {
                            "job_id": key[0],
                            "pilot_stratum": row_dict.get("pilot_stratum", ""),
                            "model_name": key[1],
                            "provider": model["provider"],
                            "model_id": model["model_id"],
                            "error_type": "interrupted_in_flight",
                            "error": error_text,
                            "attempts": attempts,
                            "input_tokens": cumulative_input_tokens,
                            "output_tokens": cumulative_output_tokens,
                            "latency_seconds": cumulative_latency,
                            "estimated_cost_usd": cumulative_cost,
                            "cost_unknown": True,
                            "failed_at_utc": now_utc(),
                        }
                        commit_terminal_checkpoint(
                            checkpoint_directory, failure_path, failures_by_key,
                            state, "failed", record,
                        )
                        done.add(key)
                        break

                    if status == "response_received":
                        last_error: Exception | None = None
                        try:
                            if state.get("usage_metadata_valid") is False:
                                raise ValueError(
                                    "Provider response is missing positive input/output usage metadata"
                                )
                            if not str(state.get("response_text", "")).strip():
                                raise ValueError("OpenAI response contained no output text")
                            extracted_prediction = extract_json(str(state["response_text"]))
                            prediction, normalization_warnings = normalize_prediction(
                                extracted_prediction
                            )
                            # Classification/logic violations remain terminal. Evidence
                            # quote mismatches and provider-added keys are retained as
                            # auditable warnings so a sound model decision is not lost.
                            errors = validate_prediction(prediction)
                            if errors:
                                raise ValueError("; ".join(errors))
                            evidence_warnings = [
                                error
                                for error in validate_prediction(
                                    prediction, source_text=posting_source
                                )
                                if "exact substring" in error
                            ]
                            validation_warnings = normalization_warnings + evidence_warnings
                        except Exception as error:
                            last_error = error

                        if last_error is None:
                            record = {
                                "job_id": key[0],
                                "pilot_stratum": row_dict.get("pilot_stratum", ""),
                                "model_name": key[1],
                                "provider": model["provider"],
                                "model_id": model["model_id"],
                                "prediction": prediction,
                                "validation_warnings": validation_warnings,
                                "evidence_substrings_valid": not evidence_warnings,
                                "provider_schema_exact": not normalization_warnings,
                                "input_tokens": cumulative_input_tokens,
                                "output_tokens": cumulative_output_tokens,
                                "latency_seconds": cumulative_latency,
                                "estimated_cost_usd": cumulative_cost,
                                "attempts": attempts,
                                "ambiguous_in_flight_attempts": int(
                                    state.get("ambiguous_in_flight_attempts", 0)
                                ),
                                "cost_unknown": bool(state.get("cost_unknown", False)),
                                "completed_at_utc": now_utc(),
                            }
                            commit_terminal_checkpoint(
                                checkpoint_directory, prediction_path, predictions_by_key,
                                state, "succeeded", record,
                            )
                            done.add(key)
                            break

                        state = {
                            **state,
                            "status": "retry_pending",
                            "last_error_type": classify_error(last_error),
                            "last_error": str(last_error),
                        }
                        save_checkpoint(checkpoint_directory, state)
                        checkpoints[key] = state
                        status = "retry_pending"
                        if attempts < max_attempts:
                            time.sleep(min(2 ** (attempts - 1), 8))

                    if status == "retry_pending" and attempts >= max_attempts:
                        record = {
                            "job_id": key[0],
                            "pilot_stratum": row_dict.get("pilot_stratum", ""),
                            "model_name": key[1],
                            "provider": model["provider"],
                            "model_id": model["model_id"],
                            "error_type": state.get("last_error_type", "provider_or_runtime"),
                            "error": state.get("last_error", "retry limit reached"),
                            "attempts": attempts,
                            "input_tokens": cumulative_input_tokens,
                            "output_tokens": cumulative_output_tokens,
                            "latency_seconds": cumulative_latency,
                            "estimated_cost_usd": cumulative_cost,
                            "cost_unknown": bool(state.get("cost_unknown", False)),
                            "ambiguous_in_flight_attempts": int(
                                state.get("ambiguous_in_flight_attempts", 0)
                            ),
                            "failed_at_utc": now_utc(),
                        }
                        commit_terminal_checkpoint(
                            checkpoint_directory, failure_path, failures_by_key,
                            state, "failed", record,
                        )
                        done.add(key)
                        break

                    if max_cost is not None and total_cost >= float(max_cost):
                        stopped_for_budget = True
                        break

                    attempt_number = attempts + 1
                    state = {
                        **state,
                        "status": "in_flight",
                        "attempts": attempt_number,
                        "request_started_at_utc": now_utc(),
                    }
                    state.pop("response_text", None)
                    state.pop("raw_record", None)
                    save_checkpoint(checkpoint_directory, state)
                    checkpoints[key] = state

                    try:
                        text, usage, raw, latency = call(
                            model["provider"],
                            model["model_id"],
                            selected_system_prompt,
                            user_prompt,
                            int(config.get("timeout_seconds", 90)),
                            int(config["max_output_tokens"]),
                            model.get("reasoning_effort"),
                        )
                    except Exception as error:
                        state = {
                            **state,
                            "status": "retry_pending",
                            "last_error_type": classify_error(error),
                            "last_error": str(error),
                        }
                        save_checkpoint(checkpoint_directory, state)
                        checkpoints[key] = state
                        if attempt_number < max_attempts:
                            time.sleep(min(2 ** (attempt_number - 1), 8))
                        continue

                    input_tokens, output_tokens = usage_tokens(model["provider"], usage)
                    usage_metadata_valid = input_tokens > 0 and output_tokens > 0
                    cost = 0.0
                    if usage_metadata_valid:
                        cost = (
                            input_tokens * float(model["input_usd_per_million"])
                            + output_tokens * float(model["output_usd_per_million"])
                        ) / 1_000_000
                        total_cost += cost
                    cumulative_input_tokens += max(input_tokens, 0)
                    cumulative_output_tokens += max(output_tokens, 0)
                    cumulative_latency += float(latency)
                    cumulative_cost += cost
                    raw_record = {
                        "job_id": key[0],
                        "model_name": key[1],
                        "attempt": attempt_number,
                        "received_at_utc": now_utc(),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_seconds": latency,
                        "estimated_cost_usd": cost,
                        "cost_unknown": not usage_metadata_valid,
                        "response_text": text,
                        "response": raw,
                    }
                    state = {
                        **state,
                        "status": "response_received",
                        "response_text": text,
                        "raw_record": raw_record,
                        "usage_metadata_valid": usage_metadata_valid,
                        "input_tokens": cumulative_input_tokens,
                        "output_tokens": cumulative_output_tokens,
                        "latency_seconds": cumulative_latency,
                        "estimated_cost_usd": cumulative_cost,
                    }
                    # Save the full received response before touching the JSONL
                    # files, so a restart can validate it without another API call.
                    save_checkpoint(checkpoint_directory, state)
                    checkpoints[key] = state
                    raw_key = raw_record_key(raw_record)
                    if raw_key not in raw_keys:
                        append_jsonl(raw_path, raw_record)
                        raw_keys.add(raw_key)
                    # The next loop iteration validates the saved response.

                update_manifest_progress(
                    manifest_path, manifest, len(predictions_by_key), len(failures_by_key),
                    total_cost, stopped_for_budget, "running",
                )
                if stopped_for_budget:
                    break
            if stopped_for_budget:
                break
    except BaseException:
        rebuild_usage(prediction_path, usage_path)
        update_manifest_progress(
            manifest_path, manifest, len(predictions_by_key), len(failures_by_key),
            total_cost, stopped_for_budget, "interrupted",
        )
        run_lock.release()
        raise

    rebuild_usage(prediction_path, usage_path)
    manifest["finished_at_utc"] = now_utc()
    final_status = "stopped_for_budget" if stopped_for_budget else "completed"
    update_manifest_progress(
        manifest_path, manifest, len(predictions_by_key), len(failures_by_key),
        total_cost, stopped_for_budget, final_status,
    )
    run_lock.release()
    print(f"Run: {run_id}")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Prompt hash: {current_prompt_hash}")
    print(f"Output: {output}")
    if stopped_for_budget:
        print(f"Stopped before the next call because the USD {float(max_cost):.2f} ceiling was reached.")


if __name__ == "__main__":
    main()
