# Local data (never committed)

Place proprietary input files in this directory. This folder is ignored by Git except for this README.

## Existing reviewed set
Expected example:

`llm_extraction_pilot_300_with_assistant_judgment.xlsx`

Required API-input columns:
- `job_id`
- `title_raw`
- `jobtitle_translated`
- `description`
- optional metadata for stratified sampling

Reference labels may be in the same file or a separate file keyed by unique `job_id`. The existing sampler restricts eligibility to IDs present in the supplied reference file. Give the labels an accurate provenance with `--reference-origin`; AI-provisional review is not human ground truth.

## Fresh candidate pool
Provide a CSV, Excel, or Parquet file containing at least the four API-input fields above. For the 50 difficult cases, add `difficult_case=true/false` or customize `src/revelio_pilot/prepare.py` to reproduce the research-defined difficult-case rule.

Before an API run, confirm that sending job-posting text to each provider is permitted. Never push Revelio raw data, job descriptions, judgments, or API responses to any Git repository.
