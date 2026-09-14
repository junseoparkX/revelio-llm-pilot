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
from .prompt import SYSTEM_PROMPT, build_user_prompt, prompt_hash
from .providers import call, required_key
from .schema import PREDICTION_SCHEMA, validate_prediction


INPUT_COLUMNS = ["job_id", "title_raw", "jobtitle_translated", "description", "pilot_stratum"]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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
        return int(usage.get("promptTokenCount", 0)), int(usage.get("candidatesTokenCount", 0))
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
        names.append(model["name"])
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique")
    max_output_tokens = config.get("max_output_tokens")
    if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        raise ValueError("Config must define a positive integer max_output_tokens")
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
    with usage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["instruction_check", "frozen_test"], required=True)
    parser.add_argument("--require-prompt-hash")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config, require_keys=not args.dry_run)
    current_prompt_hash = prompt_hash()
    schema_hash = hashlib.sha256(
        json.dumps(PREDICTION_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
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

    manifest = {
        "run_id": run_id,
        "phase": args.phase,
        "input": str(input_path),
        "input_sha256": input_hash,
        "selected_jobs_sha256": selected_jobs_hash,
        "selected_rows": len(frame),
        "prompt_sha256": current_prompt_hash,
        "schema_sha256": schema_hash,
        "models": config["models"],
        "seed": config.get("seed"),
        "max_cost_usd": config.get("max_cost_usd"),
        "max_retries": config.get("max_retries", 0),
        "max_output_tokens": config["max_output_tokens"],
        "package_version": __version__,
        "started_at_utc": now_utc(),
        "expected_calls": expected_calls,
    }

    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "phase", "input_sha256", "selected_jobs_sha256", "selected_rows", "prompt_sha256",
            "schema_sha256", "models", "max_retries", "max_output_tokens", "expected_calls",
        ):
            if existing_manifest.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume run {run_id}: manifest mismatch for {key}")
        manifest = existing_manifest
    else:
        (output / "prompts").mkdir(parents=True, exist_ok=False)
        (output / "prompts" / "prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
        (output / "prompts" / "schema.json").write_text(
            json.dumps(PREDICTION_SCHEMA, indent=2), encoding="utf-8"
        )
        write_json(manifest_path, manifest)

    predictions = load_jsonl(prediction_path)
    failures = load_jsonl(failure_path)
    done = {
        (str(record["job_id"]), record["model_name"])
        for record in predictions + failures
    }
    total_cost = sum(float(record.get("estimated_cost_usd", 0)) for record in load_jsonl(raw_path))
    max_cost = config.get("max_cost_usd")
    stopped_for_budget = False

    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        user_prompt = build_user_prompt(row_dict)
        posting_source = "\n".join(
            str(row_dict.get(column, "")) for column in ("title_raw", "jobtitle_translated", "description")
        )
        for model in config["models"]:
            key = (str(row_dict["job_id"]), model["name"])
            if key in done:
                continue
            if max_cost is not None and total_cost >= float(max_cost):
                stopped_for_budget = True
                break

            last_error: Exception | None = None
            attempts = 0
            cumulative_input_tokens = 0
            cumulative_output_tokens = 0
            cumulative_latency = 0.0
            cumulative_cost = 0.0
            for attempt in range(int(config.get("max_retries", 0)) + 1):
                if max_cost is not None and total_cost >= float(max_cost):
                    stopped_for_budget = True
                    last_error = RuntimeError("cost ceiling reached before the next attempt")
                    break
                attempts = attempt + 1
                try:
                    text, usage, raw, latency = call(
                        model["provider"],
                        model["model_id"],
                        SYSTEM_PROMPT,
                        user_prompt,
                        int(config.get("timeout_seconds", 90)),
                        int(config["max_output_tokens"]),
                    )
                    input_tokens, output_tokens = usage_tokens(model["provider"], usage)
                    if input_tokens <= 0 or output_tokens <= 0:
                        raise ValueError("Provider response is missing positive input/output usage metadata")
                    cost = (
                        input_tokens * float(model["input_usd_per_million"])
                        + output_tokens * float(model["output_usd_per_million"])
                    ) / 1_000_000
                    cumulative_input_tokens += input_tokens
                    cumulative_output_tokens += output_tokens
                    cumulative_latency += latency
                    cumulative_cost += cost
                    total_cost += cost
                    append_jsonl(raw_path, {
                        "job_id": str(row_dict["job_id"]), "model_name": model["name"],
                        "attempt": attempts, "received_at_utc": now_utc(),
                        "input_tokens": input_tokens, "output_tokens": output_tokens,
                        "latency_seconds": latency, "estimated_cost_usd": cost, "response": raw,
                    })
                    prediction = extract_json(text)
                    errors = validate_prediction(prediction, source_text=posting_source)
                    if errors:
                        raise ValueError("; ".join(errors))
                    record = {
                        "job_id": str(row_dict["job_id"]),
                        "pilot_stratum": row_dict.get("pilot_stratum", ""),
                        "model_name": model["name"],
                        "provider": model["provider"],
                        "model_id": model["model_id"],
                        "prediction": prediction,
                        "input_tokens": cumulative_input_tokens,
                        "output_tokens": cumulative_output_tokens,
                        "latency_seconds": cumulative_latency,
                        "estimated_cost_usd": cumulative_cost,
                        "attempts": attempts,
                        "completed_at_utc": now_utc(),
                    }
                    append_jsonl(prediction_path, record)
                    done.add(key)
                    last_error = None
                    break
                except Exception as error:
                    last_error = error
                    if attempt < int(config.get("max_retries", 0)):
                        time.sleep(min(2**attempt, 8))

            if last_error is not None:
                append_jsonl(
                    failure_path,
                    {
                        "job_id": str(row_dict["job_id"]),
                        "pilot_stratum": row_dict.get("pilot_stratum", ""),
                        "model_name": model["name"],
                        "provider": model["provider"],
                        "model_id": model["model_id"],
                        "error_type": classify_error(last_error),
                        "error": str(last_error),
                        "attempts": attempts,
                        "input_tokens": cumulative_input_tokens,
                        "output_tokens": cumulative_output_tokens,
                        "latency_seconds": cumulative_latency,
                        "estimated_cost_usd": cumulative_cost,
                        "failed_at_utc": now_utc(),
                    },
                )
                done.add(key)
            if stopped_for_budget:
                break
        if stopped_for_budget:
            break

    rebuild_usage(prediction_path, usage_path)
    manifest.update(
        {
            "finished_at_utc": now_utc(),
            "successful_calls": len(load_jsonl(prediction_path)),
            "terminal_failures_recorded": len(load_jsonl(failure_path)),
            "estimated_cost_usd": total_cost,
            "stopped_for_budget": stopped_for_budget,
        }
    )
    write_json(manifest_path, manifest)
    print(f"Run: {run_id}")
    print(f"Prompt hash: {current_prompt_hash}")
    print(f"Output: {output}")
    if stopped_for_budget:
        print(f"Stopped before the next call because the USD {float(max_cost):.2f} ceiling was reached.")


if __name__ == "__main__":
    main()
