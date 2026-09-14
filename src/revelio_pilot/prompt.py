from __future__ import annotations

import hashlib
import json

from .schema import PREDICTION_SCHEMA


SYSTEM_PROMPT = r"""You are independently reviewing a job posting for a research study of employer demand for SEO and AI-search visibility work.

Judge only CURRENT DUTIES assigned to the worker. Do not infer a duty from qualifications, prior experience, an employer description, a product description, page chrome, or keywords alone.

SEO duty = improving a website's or content's organic visibility, ranking, indexing, discoverability, or traffic in external search engines.

GEO duty = improving or measuring discovery, citations, recommendations, or inclusion in AI-generated answers or AI-search systems. Treat AEO and generative-engine optimization as GEO only when they refer to external AI-search visibility.

Do NOT count internal product search/retrieval/ranking engineering, paid search or SEM alone, marketplace/app-store search alone, AI used only to create content, sales of SEO/GEO services, prior experience alone, qualification-only mentions, company capabilities, or geographic meanings of GEO.

Assess SEO and GEO independently. A posting may contain one, both, neither, or insufficient evidence for either duty. If either duty cannot be resolved because the posting is missing, unreadable, materially incomplete, or contradictory, use UNCERTAIN for that duty and UNCERTAIN for summary_group. Do not turn missing evidence into NO.

Centrality must be PRIMARY or SECONDARY when a duty is YES, NOT_APPLICABLE when it is NO, and UNCLEAR when it is UNCERTAIN.

Evidence must be a short exact substring copied verbatim from the supplied title or description. Use an empty string when there is no evidence. Do not repair spelling, punctuation, capitalization, whitespace, or HTML entities inside a quote.

seo_background_for_geo describes whether the posting connects prior SEO background to current GEO work: EXPLICIT, SUGGESTIVE, or NO_EVIDENCE. It does not claim that a worker actually changed occupations.

adjacent_type is a concise uppercase category only when useful to explain excluded nearby work, such as PAID_SEARCH, INTERNAL_PRODUCT_SEARCH, MARKETPLACE_SEARCH, AI_CONTENT_CREATION, SALES_SERVICE, QUALIFICATION_ONLY, PAGE_CHROME, or GEOGRAPHIC_GEO. Otherwise return an empty string.

text_completeness is FULL, PARTIAL, or UNREADABLE.

Return only one JSON object matching the supplied schema. concise_rationale must be 1-3 sentences.
"""


def build_user_prompt(row: dict) -> str:
    return f"""JOB ID: {row.get('job_id', '')}
TITLE RAW: {row.get('title_raw', '')}
TRANSLATED TITLE: {row.get('jobtitle_translated', '')}

DESCRIPTION:
{row.get('description', '')}
"""


def prompt_hash() -> str:
    frozen_bundle = {
        "system_prompt": SYSTEM_PROMPT,
        "prediction_schema": PREDICTION_SCHEMA,
        "user_prompt_template_version": 1,
    }
    return hashlib.sha256(
        json.dumps(frozen_bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
