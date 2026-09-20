from __future__ import annotations

from typing import Any


ALLOWED = {"YES", "NO", "UNCERTAIN"}
CENTRALITY = {"PRIMARY", "SECONDARY", "NOT_APPLICABLE", "UNCLEAR"}
GROUPS = {"SEO_ONLY", "GEO_ONLY", "BOTH", "NEITHER", "UNCERTAIN"}
BACKGROUND = {"EXPLICIT", "SUGGESTIVE", "NO_EVIDENCE"}
COMPLETENESS = {"FULL", "PARTIAL", "UNREADABLE"}

REQUIRED_FIELDS = (
    "seo_duty",
    "geo_duty",
    "seo_centrality",
    "geo_centrality",
    "seo_evidence",
    "geo_evidence",
    "required_prior_experience",
    "prior_experience_evidence",
    "seo_background_for_geo",
    "adjacent_type",
    "text_completeness",
    "uncertainty_reason",
    "summary_group",
    "concise_rationale",
)


PREDICTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "seo_duty": {"type": "string", "enum": sorted(ALLOWED)},
        "geo_duty": {"type": "string", "enum": sorted(ALLOWED)},
        "seo_centrality": {"type": "string", "enum": sorted(CENTRALITY)},
        "geo_centrality": {"type": "string", "enum": sorted(CENTRALITY)},
        "seo_evidence": {"type": "string"},
        "geo_evidence": {"type": "string"},
        "required_prior_experience": {"type": "string"},
        "prior_experience_evidence": {"type": "string"},
        "seo_background_for_geo": {"type": "string", "enum": sorted(BACKGROUND)},
        "adjacent_type": {"type": "string"},
        "text_completeness": {"type": "string", "enum": sorted(COMPLETENESS)},
        "uncertainty_reason": {"type": "string"},
        "summary_group": {"type": "string", "enum": sorted(GROUPS)},
        "concise_rationale": {"type": "string"},
    },
    "required": list(REQUIRED_FIELDS),
}


def expected_group(seo_duty: str, geo_duty: str) -> str:
    if "UNCERTAIN" in {seo_duty, geo_duty}:
        return "UNCERTAIN"
    if seo_duty == "YES" and geo_duty == "YES":
        return "BOTH"
    if seo_duty == "YES":
        return "SEO_ONLY"
    if geo_duty == "YES":
        return "GEO_ONLY"
    return "NEITHER"


def validate_prediction(obj: Any, source_text: str | None = None) -> list[str]:
    if not isinstance(obj, dict):
        return ["prediction must be a JSON object"]

    errors: list[str] = []
    missing = [key for key in REQUIRED_FIELDS if key not in obj]
    extra = sorted(set(obj) - set(REQUIRED_FIELDS))
    if missing:
        errors.append(f"missing fields: {missing}")
    if extra:
        errors.append(f"unexpected fields: {extra}")

    for key in ("seo_duty", "geo_duty"):
        if obj.get(key) not in ALLOWED:
            errors.append(f"{key} must be one of {sorted(ALLOWED)}")
    for key in ("seo_centrality", "geo_centrality"):
        if obj.get(key) not in CENTRALITY:
            errors.append(f"{key} must be one of {sorted(CENTRALITY)}")
    if obj.get("summary_group") not in GROUPS:
        errors.append(f"summary_group must be one of {sorted(GROUPS)}")
    if obj.get("seo_background_for_geo") not in BACKGROUND:
        errors.append(f"seo_background_for_geo must be one of {sorted(BACKGROUND)}")
    if obj.get("text_completeness") not in COMPLETENESS:
        errors.append(f"text_completeness must be one of {sorted(COMPLETENESS)}")

    for key in REQUIRED_FIELDS:
        if key in obj and not isinstance(obj[key], str):
            errors.append(f"{key} must be a string")

    for prefix in ("seo", "geo"):
        duty = obj.get(f"{prefix}_duty")
        centrality = obj.get(f"{prefix}_centrality")
        evidence = obj.get(f"{prefix}_evidence")
        if duty == "YES":
            if centrality not in {"PRIMARY", "SECONDARY"}:
                errors.append(f"{prefix}_centrality must be PRIMARY or SECONDARY when duty is YES")
            if not evidence:
                errors.append(f"{prefix}_evidence is required when duty is YES")
        elif duty == "NO":
            if centrality != "NOT_APPLICABLE":
                errors.append(f"{prefix}_centrality must be NOT_APPLICABLE when duty is NO")
            if evidence:
                errors.append(f"{prefix}_evidence must be empty when duty is NO")
        elif duty == "UNCERTAIN" and centrality != "UNCLEAR":
            errors.append(f"{prefix}_centrality must be UNCLEAR when duty is UNCERTAIN")

    seo = obj.get("seo_duty")
    geo = obj.get("geo_duty")
    if seo in ALLOWED and geo in ALLOWED:
        derived = expected_group(seo, geo)
        if obj.get("summary_group") != derived:
            errors.append(f"summary_group must be {derived} for seo_duty={seo}, geo_duty={geo}")

    if "UNCERTAIN" in {seo, geo} and not obj.get("uncertainty_reason"):
        errors.append("uncertainty_reason is required when either duty is UNCERTAIN")

    if source_text is not None:
        for key in ("seo_evidence", "geo_evidence", "prior_experience_evidence"):
            quote = obj.get(key)
            if isinstance(quote, str) and quote and quote not in source_text:
                errors.append(f"{key} is not an exact substring of the supplied source text")

    return errors


def normalize_prediction(obj: Any) -> tuple[Any, list[str]]:
    """Drop provider-added keys while preserving the raw response separately.

    Some structured-output implementations return every required field correctly
    but also invent a symmetric field that is not part of the research schema.
    The runner retains the provider response verbatim in raw_responses.jsonl and
    records this deterministic normalization as a validation warning.
    """
    if not isinstance(obj, dict):
        return obj, []
    extra = sorted(set(obj) - set(REQUIRED_FIELDS))
    cleaned = {key: obj[key] for key in REQUIRED_FIELDS if key in obj}
    warnings = [f"removed unexpected fields: {extra}"] if extra else []
    return cleaned, warnings
