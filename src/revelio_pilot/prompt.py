from __future__ import annotations

import hashlib
import json

from .schema import PREDICTION_SCHEMA


BASELINE_PROMPT_V1 = r"""You are independently reviewing a job posting for a research study of employer demand for SEO and AI-search visibility work.

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


IMPROVED_PROMPT_V2 = r"""You are independently reviewing a job posting for a research study of employer demand for SEO and AI-search visibility work. Classify assigned work, not keyword presence.

Use the supplied title and description together.

1. Check text completeness first.
- FULL: the operative duties are present and readable, even if boilerplate is omitted.
- PARTIAL: the text itself is cut off or omits a material duties section, so at least one duty cannot be resolved.
- UNREADABLE: blank, corrupted, or not meaningfully interpretable.
Do not infer truncation from length alone. If missing or contradictory text prevents a decision for either duty, mark that duty UNCERTAIN and make summary_group UNCERTAIN. Do not turn absent evidence in incomplete text into NO.

2. Attribute the work to the worker.
Count only a current action, ownership area, deliverable, or measured outcome assigned to this role. Do not infer a duty from prior experience, qualifications, preferred skills, employer capabilities, product descriptions, page chrome, or a keyword alone. A title can support a duty only when it clearly names the function and the description is compatible; an unexplained acronym in weak or unrelated context is not enough.

3. Judge SEO and GEO independently.
- SEO duty: improve or measure a website's or content's organic visibility, ranking, indexing, discoverability, or traffic in external conventional search engines. Strong cues include organic search, Google/SERP rankings, crawling/indexing, technical SEO, keyword optimization, and SEO performance.
- GEO duty: improve or measure public content's discovery, citation, recommendation, or inclusion in external AI-generated answers or AI-search systems. Strong cues include AI-generated answers, answer engines, LLM discoverability, AI-powered search results, citation/share-of-voice in systems such as ChatGPT or Perplexity, and structuring public content for those systems.
- AEO, GEO, or generative-engine optimization counts as GEO when it is an assigned workstream in an external search/content visibility context. If the acronym's meaning or external target remains materially ambiguous, use UNCERTAIN rather than assuming YES.
- Generic "search," discoverability, personalization, recommendations, or content consumption does not establish SEO or GEO without the relevant external target.

4. Apply exclusions by context.
Do not count internal product search, retrieval, ranking, RAG, recommendation, or model engineering; paid search or SEM alone; marketplace/app-store search alone; AI used only to create content; sales or promotion of SEO/GEO services without ownership of delivery or results; qualification-only mentions; company capabilities; or geographic meanings of GEO. Managing or executing client SEO/GEO delivery can count when the role owns that work or its outcomes.

5. Complete the fields consistently.
- For a YES duty, centrality is PRIMARY or SECONDARY and evidence is a short exact substring that contains the assigned action and relevant target when possible.
- For a NO duty, centrality is NOT_APPLICABLE and evidence is an empty string.
- For an UNCERTAIN duty, centrality is UNCLEAR and uncertainty_reason names the missing or ambiguous fact. Evidence may quote the ambiguous phrase exactly, or be empty when the problem is missing text.
- Evidence must be copied verbatim from the supplied title or description. Do not repair spelling, punctuation, capitalization, whitespace, or HTML entities.
- summary_group is derived from the two duty decisions: BOTH, SEO_ONLY, GEO_ONLY, NEITHER, or UNCERTAIN when either duty is UNCERTAIN.
- required_prior_experience summarizes explicit required or preferred prior work experience; use NO when none is stated. prior_experience_evidence is one exact quote, or an empty string when none is stated.
- seo_background_for_geo is EXPLICIT or SUGGESTIVE only when the posting connects prior SEO background to current GEO work; otherwise NO_EVIDENCE.
- adjacent_type is one concise uppercase exclusion category when useful, such as PAID_SEARCH, INTERNAL_PRODUCT_SEARCH, INTERNAL_AI_PRODUCT, MARKETPLACE_SEARCH, AI_CONTENT_CREATION, SALES_SERVICE, QUALIFICATION_ONLY, PAGE_CHROME, or GEOGRAPHIC_GEO; otherwise use an empty string.

Return only one JSON object matching the supplied schema. concise_rationale must contain exactly two labeled sentences in this order: "SEO: <YES|NO|UNCERTAIN> — <why>. GEO: <YES|NO|UNCERTAIN> — <why>." For YES, name the assigned action and external target. For NO, name the decisive exclusion or state that the complete posting assigns no such external-visibility work. For UNCERTAIN, name the exact ambiguity or missing section. Explain the decision; do not merely repeat the label or evidence quote.
"""


IMPROVED_PROMPT_V2_1 = IMPROVED_PROMPT_V2 + r"""

V2.1 FIRST-PASS SCREENING CLARIFICATIONS
These clarifications control when they are more specific than the rules above. This is a high-recall first pass: a credible but unresolved current-duty signal should remain UNCERTAIN for later review, while a complete posting with no credible signal should still be NO.

1. Conventional workstream names.
- In a current responsibility, ownership area, deliverable, or team remit, the unqualified terms SEO, search engine optimization, AEO, answer engine optimization, GEO, and generative engine optimization conventionally name external visibility work.
- When one of those terms is explicitly assigned as current work and the posting does not redefine it as an excluded activity, classify the corresponding duty YES even if the same sentence does not restate the external target.
- An acronym in a title alone, in keyword stuffing, or in an unrelated sentence is not sufficient. If the current-duty context is credible but the acronym's meaning is genuinely unresolved, use UNCERTAIN rather than NO.

2. Current work versus worker background.
- Keep current duties separate from required or preferred experience. Prior SEO experience does not itself create a current SEO duty, and prior AI/GEO experience does not itself create a current GEO duty.
- A posting may therefore be GEO YES and SEO NO while separately recording SEO as required prior experience.

3. Ownership and adjacent activity.
- Count hands-on execution, strategy ownership, optimization, testing, measurement, or accountable client delivery of SEO/GEO outcomes.
- Do not count merely coordinating projects, staffing specialists, selling services, finding sales prospects, reading search-visibility signals for outreach, or reporting generic marketing results unless the worker is also responsible for improving or delivering the relevant organic-search or AI-search outcome.
- Mentioning that an employer or client offers SEO/GEO services is not a duty for this worker.

4. External visibility boundary.
- Internal knowledge bases, enterprise search, RAG, agent retrieval, internal LLM consumption, product search, and recommendation engineering are NO for GEO unless the role separately owns public-content visibility in external AI answers or AI-search systems.
- Generic AI, LLM, discovery, "traditional search," or content-optimization language does not establish SEO or GEO without either a conventional workstream name or a concrete external-visibility action.

5. Evidence and explanation.
- Prefer one short, contiguous, verbatim quote that most directly proves each YES or UNCERTAIN decision. Preserve the source exactly, including HTML entities and whitespace; do not combine noncontiguous fragments.
- Keep each of the two rationale sentences brief. State the decisive assigned action or exclusion without adding facts not present in the posting.
"""


DEFAULT_PROMPT_VERSION = "baseline_v1"
PROMPT_VERSIONS = {
    "baseline_v1": BASELINE_PROMPT_V1,
    "improved_v2": IMPROVED_PROMPT_V2,
    "improved_v2_1": IMPROVED_PROMPT_V2_1,
}

# Backward-compatible alias used by the already-completed pilot and P1 tooling.
SYSTEM_PROMPT = BASELINE_PROMPT_V1


def get_system_prompt(version: str = DEFAULT_PROMPT_VERSION) -> str:
    try:
        return PROMPT_VERSIONS[version]
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPT_VERSIONS))
        raise ValueError(f"Unknown prompt version {version!r}; choose one of: {choices}") from exc


def build_user_prompt(row: dict) -> str:
    return f"""JOB ID: {row.get('job_id', '')}
TITLE RAW: {row.get('title_raw', '')}
TRANSLATED TITLE: {row.get('jobtitle_translated', '')}

DESCRIPTION:
{row.get('description', '')}
"""


def prompt_hash(system_prompt: str | None = None) -> str:
    selected_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    frozen_bundle = {
        "system_prompt": selected_prompt,
        "prediction_schema": PREDICTION_SCHEMA,
        "user_prompt_template_version": 1,
    }
    return hashlib.sha256(
        json.dumps(frozen_bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
