from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from .schema import validate_prediction


LABEL_ORDER = ["YES", "NO", "UNCERTAIN"]
GROUP_ORDER = ["SEO_ONLY", "GEO_ONLY", "BOTH", "NEITHER", "UNCERTAIN"]


def load_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    source = Path(path)
    if optional and not source.exists():
        return []
    records = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {source} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {source} line {line_number}")
        records.append(value)
    return records


def load_table(path: str, sheet: str | None = None) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=sheet or 0, dtype={"job_id": str})
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
        frame["job_id"] = frame["job_id"].astype(str)
        return frame
    return pd.read_csv(source, dtype={"job_id": str})


def parse_reference(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, TypeError):
            pass
    return {}


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(true: pd.Series, predicted: pd.Series) -> dict[str, float]:
    pairs = [(str(t), str(p)) for t, p in zip(true, predicted) if pd.notna(t) and pd.notna(p)]
    true_positive = sum(t == "YES" and p == "YES" for t, p in pairs)
    false_positive = sum(t != "YES" and p == "YES" for t, p in pairs)
    false_negative = sum(t == "YES" and p != "YES" for t, p in pairs)
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


def accuracy(true: pd.Series, predicted: pd.Series) -> float:
    pairs = [(str(t), str(p)) for t, p in zip(true, predicted) if pd.notna(t) and pd.notna(p)]
    return safe_div(sum(t == p for t, p in pairs), len(pairs))


def latest_by_key(records: list[dict]) -> dict[tuple[str, str], dict]:
    latest = {}
    for record in records:
        key = (str(record.get("job_id", "")), str(record.get("model_name", "")))
        latest[key] = record
    return latest


def evidence_is_valid(quote: str, source_text: str | None) -> bool | None:
    if not quote:
        return None
    if source_text is None:
        return None
    return quote in source_text


def metric_row(group: pd.DataFrame, model_name: str, stratum: str | None = None) -> dict:
    seo = binary_metrics(group["seo_true"], group["seo_pred"])
    geo = binary_metrics(group["geo_true"], group["geo_pred"])
    row = {
        "model_name": model_name,
        "n_scored": len(group),
        "seo_precision": seo["precision"],
        "seo_recall": seo["recall"],
        "seo_f1": seo["f1"],
        "geo_precision": geo["precision"],
        "geo_recall": geo["recall"],
        "geo_f1": geo["f1"],
        "seo_accuracy": accuracy(group["seo_true"], group["seo_pred"]),
        "geo_accuracy": accuracy(group["geo_true"], group["geo_pred"]),
        "group_accuracy": accuracy(group["group_true"], group["group_pred"]),
        "estimated_cost_usd": group["estimated_cost_usd"].sum(),
        "mean_latency_seconds": group["latency_seconds"].mean(),
        "mean_attempts": group["attempts"].mean(),
        "possible_duty_prior_confusions": int(group["possible_duty_prior_confusion"].sum()),
    }
    for prefix in ("seo", "geo"):
        valid = group[f"{prefix}_evidence_valid"].dropna()
        row[f"{prefix}_evidence_substring_rate"] = valid.mean() if len(valid) else None
    if stratum is not None:
        row["pilot_stratum"] = stratum
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures")
    parser.add_argument("--source")
    parser.add_argument("--source-sheet")
    args = parser.parse_args()

    predictions = latest_by_key(load_jsonl(args.predictions))
    failures = latest_by_key(load_jsonl(args.failures, optional=True)) if args.failures else {}
    references = {}
    origins = {}
    for record in load_jsonl(args.reference):
        job_id = str(record["job_id"])
        references[job_id] = parse_reference(record.get("reference"))
        origins[job_id] = record.get("reference_origin", "")

    source_text = {}
    if args.source:
        source = load_table(args.source, args.source_sheet).fillna("")
        if "job_id" not in source.columns or "description" not in source.columns:
            raise ValueError("Source table must contain job_id and description")
        if source["job_id"].duplicated().any():
            raise ValueError("job_id must be unique in the source table")
        for _, record in source.iterrows():
            source_text[str(record["job_id"])] = "\n".join(
                str(record.get(column, "")) for column in ("title_raw", "jobtitle_translated", "description")
            )

    rows = []
    for (job_id, model_name), record in predictions.items():
        reference = references.get(job_id)
        if not reference:
            continue
        prediction = record.get("prediction", {})
        combined_source = source_text.get(job_id)
        seo_valid = evidence_is_valid(str(prediction.get("seo_evidence", "")), combined_source)
        geo_valid = evidence_is_valid(str(prediction.get("geo_evidence", "")), combined_source)
        prior_reference = str(reference.get("required_prior_experience", "")).strip()
        duty_prior_proxy = (
            prior_reference != ""
            and (
                (prediction.get("seo_duty") == "YES" and reference.get("seo_duty") == "NO")
                or (prediction.get("geo_duty") == "YES" and reference.get("geo_duty") == "NO")
            )
        )
        rows.append(
            {
                "job_id": job_id,
                "reference_origin": origins.get(job_id, ""),
                "pilot_stratum": record.get("pilot_stratum", ""),
                "model_name": model_name,
                "seo_true": reference.get("seo_duty"),
                "seo_pred": prediction.get("seo_duty"),
                "geo_true": reference.get("geo_duty"),
                "geo_pred": prediction.get("geo_duty"),
                "group_true": reference.get("summary_group"),
                "group_pred": prediction.get("summary_group"),
                "seo_evidence": prediction.get("seo_evidence", ""),
                "geo_evidence": prediction.get("geo_evidence", ""),
                "seo_evidence_valid": seo_valid,
                "geo_evidence_valid": geo_valid,
                "possible_duty_prior_confusion": duty_prior_proxy,
                "input_tokens": record.get("input_tokens", 0),
                "output_tokens": record.get("output_tokens", 0),
                "latency_seconds": record.get("latency_seconds", 0),
                "estimated_cost_usd": record.get("estimated_cost_usd", 0),
                "attempts": record.get("attempts", 1),
            }
        )

    columns = [
        "job_id", "reference_origin", "pilot_stratum", "model_name", "seo_true", "seo_pred",
        "geo_true", "geo_pred", "group_true", "group_pred", "seo_evidence", "geo_evidence",
        "seo_evidence_valid", "geo_evidence_valid", "possible_duty_prior_confusion",
        "input_tokens", "output_tokens", "latency_seconds", "estimated_cost_usd", "attempts",
    ]
    scored = pd.DataFrame(rows, columns=columns)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    reference_quality = []
    for job_id, reference in sorted(references.items()):
        schema_errors = validate_prediction(reference)
        all_errors = validate_prediction(reference, source_text.get(job_id)) if args.source else schema_errors
        evidence_errors = [error for error in all_errors if "exact substring" in error]
        reference_quality.append(
            {
                "job_id": job_id,
                "reference_origin": origins.get(job_id, ""),
                "schema_or_logic_valid": not schema_errors,
                "evidence_substrings_valid": None if not args.source else not evidence_errors,
                "schema_or_logic_errors": " | ".join(schema_errors),
                "evidence_errors": " | ".join(evidence_errors),
            }
        )
    reference_quality_frame = pd.DataFrame(reference_quality)
    reference_quality_frame.to_csv(output / "reference_quality.csv", index=False)

    per_model = []
    per_stratum = []
    if not scored.empty:
        for model_name, group in scored.groupby("model_name", sort=True):
            per_model.append(metric_row(group, model_name))
        for (model_name, stratum), group in scored.groupby(["model_name", "pilot_stratum"], sort=True):
            per_stratum.append(metric_row(group, model_name, stratum))
    pd.DataFrame(per_model).to_csv(output / "per_model_metrics.csv", index=False)
    pd.DataFrame(per_stratum).to_csv(output / "per_stratum_metrics.csv", index=False)
    pd.DataFrame(per_model).to_csv(output / "summary.csv", index=False)

    disagreements = scored[
        (scored["seo_true"] != scored["seo_pred"])
        | (scored["geo_true"] != scored["geo_pred"])
        | (scored["group_true"] != scored["group_pred"])
    ] if not scored.empty else scored
    disagreements.to_csv(output / "disagreements.csv", index=False)

    confusion_rows = []
    for model_name, group in scored.groupby("model_name", sort=True):
        for field, labels in (("seo", LABEL_ORDER), ("geo", LABEL_ORDER), ("group", GROUP_ORDER)):
            for true_label in labels:
                for predicted_label in labels:
                    count = int(
                        ((group[f"{field}_true"] == true_label) & (group[f"{field}_pred"] == predicted_label)).sum()
                    )
                    confusion_rows.append(
                        {
                            "model_name": model_name,
                            "field": field,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": count,
                        }
                    )
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrices.csv", index=False)

    model_names = sorted({key[1] for key in predictions} | {key[1] for key in failures})
    call_summary = []
    for model_name in model_names:
        success_keys = {key for key in predictions if key[1] == model_name}
        failure_keys = {key for key in failures if key[1] == model_name and key not in success_keys}
        invalid_keys = {
            key
            for key in failure_keys
            if failures[key].get("error_type") in {"invalid_json", "schema_or_semantic_validation"}
        }
        attempted = len(success_keys) + len(failure_keys)
        call_summary.append(
            {
                "model_name": model_name,
                "successful_calls": len(success_keys),
                "terminal_failed_calls": len(failure_keys),
                "terminal_invalid_output_calls": len(invalid_keys),
                "failed_call_rate": safe_div(len(failure_keys), attempted),
                "invalid_output_rate": safe_div(len(invalid_keys), attempted),
            }
        )
    pd.DataFrame(call_summary).to_csv(output / "call_quality.csv", index=False)

    summary = {
        "rows_scored": len(scored),
        "reference_jobs": len(references),
        "models": model_names,
        "source_text_supplied_for_evidence_check": bool(args.source),
        "prediction_records_after_deduplication": len(predictions),
        "terminal_failure_records_after_deduplication": len(failures),
        "reference_schema_or_logic_invalid": int(
            (~reference_quality_frame["schema_or_logic_valid"]).sum()
        ) if not reference_quality_frame.empty else 0,
        "reference_evidence_substring_invalid": int(
            (reference_quality_frame["evidence_substrings_valid"] == False).sum()
        ) if args.source and not reference_quality_frame.empty else None,
        "possible_duty_prior_confusion_is_proxy": True,
        "notes": [
            "Evidence substring validity checks literal occurrence only; it does not prove semantic support.",
            "Possible duty/prior-experience confusion is a review flag, not a definitive semantic error.",
            "Precision and recall treat YES as the positive class and NO/UNCERTAIN as non-positive.",
        ],
        "call_quality": call_summary,
    }
    (output / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote evaluation to {output}")


if __name__ == "__main__":
    main()
