from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from .schema import REQUIRED_FIELDS


INPUT_COLS = ["job_id", "title_raw", "jobtitle_translated", "description"]


def load(path: str, sheet: str | None = None) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=sheet or 0, dtype={"job_id": str})
    if suffix == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source, dtype={"job_id": str})
    return frame


def ensure(frame: pd.DataFrame) -> None:
    missing = [column for column in INPUT_COLS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["job_id"].isna().any() or frame["job_id"].astype(str).str.strip().eq("").any():
        raise ValueError("job_id must be non-empty")
    if frame["job_id"].astype(str).duplicated().any():
        raise ValueError("job_id must be unique in the candidate input")


def _jsonable(value):
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return value


def _parse_reference(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("Reference column values must be JSON objects or object-like strings")
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, TypeError):
            pass
    raise ValueError("Could not parse a reference value as an object")


def _write_references(sample: pd.DataFrame, args: argparse.Namespace) -> None:
    reference_frame = sample
    if args.reference_input:
        external = load(args.reference_input, args.reference_sheet)
        if "job_id" not in external.columns:
            raise ValueError("Reference input must contain job_id")
        if external["job_id"].astype(str).duplicated().any():
            raise ValueError("job_id must be unique in the reference input")
        keep = ["job_id"] + [column for column in REQUIRED_FIELDS if column in external.columns]
        if args.reference_col in external.columns:
            keep.append(args.reference_col)
        reference_frame = sample[["job_id"]].merge(external[keep], on="job_id", how="left", validate="one_to_one")

    records = []
    for _, row in reference_frame.iterrows():
        if args.reference_col in reference_frame.columns and not pd.isna(row[args.reference_col]):
            reference = _parse_reference(row[args.reference_col])
        else:
            available = [field for field in REQUIRED_FIELDS if field in reference_frame.columns]
            if not available:
                raise ValueError(
                    "No reference labels found. Supply --reference-input, a reference object column, "
                    "or columns matching the prediction schema."
                )
            reference = {field: _jsonable(row[field]) for field in available}
        records.append(
            {
                "job_id": str(row["job_id"]),
                "reference_origin": args.reference_origin,
                "reference": reference,
            }
        )

    output = Path(args.reference_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sample_existing(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n > len(frame):
        raise ValueError(f"n must be between 1 and {len(frame)}")
    strata = [column for column in ("summary_group", "sample_stratum", "filter_priority") if column in frame.columns]
    if not strata:
        return frame.sample(n=n, random_state=seed)

    key = strata[0]
    groups = list(frame.groupby(key, dropna=False, sort=True))
    allocations = []
    for label, group in groups:
        exact = n * len(group) / len(frame)
        allocations.append([label, group, min(len(group), int(exact)), exact - int(exact)])
    remaining = n - sum(item[2] for item in allocations)
    for item in sorted(allocations, key=lambda value: (-value[3], str(value[0]))):
        if remaining == 0:
            break
        if item[2] < len(item[1]):
            item[2] += 1
            remaining -= 1

    parts = [group.sample(n=take, random_state=seed) for _, group, take, _ in allocations if take]
    sample = pd.concat(parts).drop_duplicates("job_id")
    if len(sample) < n:
        remainder = frame[~frame["job_id"].isin(sample["job_id"])].sample(n - len(sample), random_state=seed)
        sample = pd.concat([sample, remainder])
    return sample.sample(frac=1, random_state=seed)


def existing(args: argparse.Namespace) -> None:
    frame = load(args.input, args.sheet)
    ensure(frame)
    if args.reference_input:
        external = load(args.reference_input, args.reference_sheet)
        if "job_id" not in external.columns:
            raise ValueError("Reference input must contain job_id")
        external["job_id"] = external["job_id"].astype(str)
        if external["job_id"].duplicated().any():
            raise ValueError("job_id must be unique in the reference input")
        reviewed_ids = set(external["job_id"])
        frame = frame[frame["job_id"].astype(str).isin(reviewed_ids)].copy()
        if frame.empty:
            raise ValueError("Candidate input and reference input have no job_id overlap")
        if "summary_group" in external.columns and "summary_group" not in frame.columns:
            frame = frame.merge(external[["job_id", "summary_group"]], on="job_id", how="left", validate="one_to_one")
    sample = _sample_existing(frame, args.n, args.seed).copy()
    sample["pilot_stratum"] = "existing_instruction_check"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample[INPUT_COLS + ["pilot_stratum"]].fillna("").to_csv(output, index=False)
    _write_references(sample, args)
    print(f"Wrote {len(sample)} blind cases to {output}")
    print(f"Wrote private references to {args.reference_output}")


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0", "yes", "no", "y", "n"})
    if unknown:
        raise ValueError(f"Unrecognized difficult_case values: {unknown[:5]}")
    return normalized.isin({"true", "1", "yes", "y"})


def new(args: argparse.Namespace) -> None:
    frame = load(args.input, args.sheet)
    ensure(frame)
    if "difficult_case" not in frame.columns:
        raise ValueError(
            "Fresh pool needs a difficult_case boolean column produced by the frozen project-specific rule."
        )
    difficult_mask = _boolean_series(frame["difficult_case"])
    difficult = frame[difficult_mask]
    random_pool = frame[~difficult_mask]
    if len(difficult) < args.difficult_n or len(random_pool) < args.random_n:
        raise ValueError("Not enough rows in one or both fresh strata")
    random_sample = random_pool.sample(args.random_n, random_state=args.seed).copy()
    random_sample["pilot_stratum"] = "new_random"
    difficult_sample = difficult.sample(args.difficult_n, random_state=args.seed).copy()
    difficult_sample["pilot_stratum"] = "new_difficult"
    output_frame = pd.concat([random_sample, difficult_sample], ignore_index=True).sample(
        frac=1, random_state=args.seed
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame[INPUT_COLS + ["pilot_stratum"]].fillna("").to_csv(output, index=False)
    print(f"Wrote {len(output_frame)} fresh blind cases to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    existing_parser = subparsers.add_parser("existing")
    existing_parser.add_argument("--input", required=True)
    existing_parser.add_argument("--sheet")
    existing_parser.add_argument("--n", type=int, default=50)
    existing_parser.add_argument("--seed", type=int, default=20260914)
    existing_parser.add_argument("--reference-input")
    existing_parser.add_argument("--reference-sheet")
    existing_parser.add_argument("--reference-col", default="assistant_final_judgment")
    existing_parser.add_argument("--reference-origin", default="reviewed_existing")
    existing_parser.add_argument(
        "--reference-output", default="outputs/private_reference/reference_labels.jsonl"
    )
    existing_parser.add_argument("--output", required=True)
    existing_parser.set_defaults(fn=existing)

    new_parser = subparsers.add_parser("new")
    new_parser.add_argument("--input", required=True)
    new_parser.add_argument("--sheet")
    new_parser.add_argument("--random-n", type=int, default=50)
    new_parser.add_argument("--difficult-n", type=int, default=50)
    new_parser.add_argument("--seed", type=int, default=20260914)
    new_parser.add_argument("--output", required=True)
    new_parser.set_defaults(fn=new)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
