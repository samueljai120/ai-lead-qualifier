"""
qualifier.py — AI-powered Lead Qualification Agent (CLS Corp).

Scrapes public company data, feeds it to an LLM via OpenRouter (DeepSeek by default),
and produces a structured 1-10 score across four dimensions:
  • Budget Likelihood  — signals of financial health & willingness to invest
  • Tech Readiness     — existing tech stack sophistication, engineering presence
  • Pain Points        — identifiable operational frictions AI/automation can solve
  • Urgency            — hiring spikes, funding rounds, competitive pressure signals

Usage:
  python qualifier.py stripe.com
  python qualifier.py "Acme Corp" --tone formal --model deepseek/deepseek-chat
  python qualifier.py stripe.com --output json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from database import (
    get_enrichment,
    get_latest_score,
    init_db,
    save_score,
    upsert_lead,
)
from enricher import EnrichmentResult, enrich_lead

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("QUALIFIER_MODEL", "deepseek/deepseek-chat")
SITE_URL = os.getenv("SITE_URL", "https://clscorp.ai")
SITE_NAME = os.getenv("SITE_NAME", "CLS Corp Lead Qualifier")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreDimension:
    score: float          # 1.0 – 10.0
    reasoning: str
    signals: list[str] = field(default_factory=list)


@dataclass
class LeadScore:
    budget_likelihood: ScoreDimension
    tech_readiness: ScoreDimension
    pain_points: ScoreDimension
    urgency: ScoreDimension
    overall: float
    overall_summary: str
    recommended_action: str          # HOT / WARM / NURTURE / PASS
    model_used: str
    scored_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        def _dim(d: ScoreDimension) -> dict[str, Any]:
            return {"score": d.score, "reasoning": d.reasoning, "signals": d.signals}

        return {
            "budget_likelihood": _dim(self.budget_likelihood),
            "tech_readiness": _dim(self.tech_readiness),
            "pain_points": _dim(self.pain_points),
            "urgency": _dim(self.urgency),
            "overall": self.overall,
            "overall_summary": self.overall_summary,
            "recommended_action": self.recommended_action,
            "model_used": self.model_used,
            "scored_at": self.scored_at,
        }


@dataclass
class LeadReport:
    identifier: str
    lead_id: int
    company_name: str
    website_url: str
    domain: str
    enrichment: EnrichmentResult
    score: LeadScore
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "lead_id": self.lead_id,
            "company_name": self.company_name,
            "website_url": self.website_url,
            "domain": self.domain,
            "enrichment": self.enrichment.to_dict(),
            "score": self.score.to_dict(),
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# AI scoring
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = dedent("""
    You are an expert B2B sales intelligence analyst at CLS Corp, an AI automation agency.
    Your task is to evaluate a company lead and score it across four dimensions.
    You MUST respond with a single, valid JSON object — no markdown fences, no commentary.

    JSON schema (strict):
    {
      "budget_likelihood": {
        "score": <float 1-10>,
        "reasoning": "<2-3 sentences>",
        "signals": ["<signal1>", "<signal2>", ...]
      },
      "tech_readiness": {
        "score": <float 1-10>,
        "reasoning": "<2-3 sentences>",
        "signals": ["<signal1>", ...]
      },
      "pain_points": {
        "score": <float 1-10>,
        "reasoning": "<2-3 sentences>",
        "signals": ["<signal1>", ...]
      },
      "urgency": {
        "score": <float 1-10>,
        "reasoning": "<2-3 sentences>",
        "signals": ["<signal1>", ...]
      },
      "overall_summary": "<3-4 sentence executive summary of the lead>",
      "recommended_action": "<one of: HOT | WARM | NURTURE | PASS>"
    }

    Scoring guide:
    - budget_likelihood: 9-10 = Series B+/enterprise with AI budget lines;
                         5-8 = growth stage, likely can allocate;
                         1-4 = bootstrapped/early with no visible budget signals
    - tech_readiness:    9-10 = modern stack (cloud, APIs, CI/CD, existing automation);
                         5-8 = partial modernisation, some legacy;
                         1-4 = legacy monolith, no API signals
    - pain_points:       9-10 = explicit ops pain, repetitive process mentions, support/data backlog;
                         5-8 = implied frictions in description;
                         1-4 = no detectable pain
    - urgency:           9-10 = hiring AI/ML roles, recent funding, explicit "scale" language;
                         5-8 = growth indicators, competitive market;
                         1-4 = stable/stagnant, no growth signals

    recommended_action:
    - HOT    = overall implied score ≥ 7.5 — reach out immediately
    - WARM   = 6.0–7.4 — nurture with a targeted touch
    - NURTURE= 4.5–5.9 — add to drip sequence
    - PASS   = < 4.5 — not a fit right now
""").strip()


def _build_user_prompt(enrichment: EnrichmentResult) -> str:
    w = enrichment.website
    s = enrichment.social
    li = enrichment.linkedin

    sections: list[str] = [f"## Lead: {enrichment.identifier}\n"]

    if w:
        sections.append(
            f"### Website Data\n"
            f"URL            : {w.url}\n"
            f"Title          : {w.title}\n"
            f"Meta Desc      : {w.description}\n"
            f"Tech Stack     : {', '.join(w.tech_stack_hints) or 'unknown'}\n"
            f"Industries     : {', '.join(w.industries) or 'unknown'}\n"
            f"Employee Hint  : {w.employee_count_hint or 'not found'}\n"
            f"Has Careers    : {w.careers_page_found}\n"
            f"Has Pricing    : {w.pricing_page_found}\n"
            f"Has Blog       : {w.blog_found}\n"
            f"Contact Email  : {w.contact_email or 'not found'}\n"
            f"Social Links   : {json.dumps(w.social_links)}\n"
            f"Error          : {w.error or 'none'}\n\n"
            f"### Website Text Snippet (first 2000 chars)\n"
            f"{w.raw_text_snippet[:2000]}\n"
        )

    if s and (s.github_org or s.twitter_handle):
        sections.append(
            f"### Social / GitHub\n"
            f"GitHub Org     : {s.github_org or 'N/A'}\n"
            f"GitHub Repos   : {s.github_public_repos}\n"
            f"GitHub Stars   : {s.github_stars_total}\n"
            f"Twitter Handle : {s.twitter_handle or 'N/A'}\n"
        )

    if li and not li.is_stub:
        sections.append(
            f"### LinkedIn\n"
            f"Company Size   : {li.company_size}\n"
            f"Industry       : {li.industry}\n"
            f"Headquarters   : {li.headquarters}\n"
            f"Founded        : {li.founded_year}\n"
            f"Specialties    : {', '.join(li.specialties)}\n"
            f"Followers      : {li.follower_count}\n"
        )

    return "\n".join(sections)


async def score_lead_with_ai(
    enrichment: EnrichmentResult,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> LeadScore:
    """Call OpenRouter/DeepSeek to score the enriched lead."""

    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Export it or add it to your .env file."
        )

    user_prompt = _build_user_prompt(enrichment)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    logger.info("Sending scoring request to OpenRouter  model=%s", model)
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

    elapsed = round((time.monotonic() - t0) * 1000)
    logger.info("AI response received  [%d ms]", elapsed)

    data = response.json()
    raw_content: str = data["choices"][0]["message"]["content"].strip()

    # Strip any accidental markdown code fences
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```[a-z]*\n?", "", raw_content)
        raw_content = re.sub(r"\n?```$", "", raw_content)

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse AI JSON response: %s\nRaw: %s", exc, raw_content[:500])
        raise ValueError(f"AI returned invalid JSON: {exc}") from exc

    def _dim(key: str) -> ScoreDimension:
        d = parsed[key]
        return ScoreDimension(
            score=float(d["score"]),
            reasoning=d.get("reasoning", ""),
            signals=d.get("signals", []),
        )

    budget = _dim("budget_likelihood")
    tech = _dim("tech_readiness")
    pain = _dim("pain_points")
    urgency = _dim("urgency")

    # Weighted overall: pain_points and urgency slightly more impactful
    weights = {"budget": 0.25, "tech": 0.20, "pain": 0.30, "urgency": 0.25}
    overall = round(
        budget.score * weights["budget"]
        + tech.score * weights["tech"]
        + pain.score * weights["pain"]
        + urgency.score * weights["urgency"],
        2,
    )

    action = parsed.get("recommended_action", "WARM")
    if action not in {"HOT", "WARM", "NURTURE", "PASS"}:
        action = "WARM"  # safe fallback

    return LeadScore(
        budget_likelihood=budget,
        tech_readiness=tech,
        pain_points=pain,
        urgency=urgency,
        overall=overall,
        overall_summary=parsed.get("overall_summary", ""),
        recommended_action=action,
        model_used=model,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def qualify_lead(
    identifier: str,
    model: str = DEFAULT_MODEL,
    force_refresh: bool = False,
) -> LeadReport:
    """
    Full qualification pipeline:
      1. Upsert lead record in DB
      2. Enrich from public sources
      3. Score with AI
      4. Persist score
      5. Return structured LeadReport
    """
    init_db()

    from urllib.parse import urlparse
    from enricher import _normalise_url

    url = _normalise_url(identifier)
    parsed = urlparse(url)
    domain = parsed.netloc.lstrip("www.")
    company_name = (
        domain.split(".")[0].title()
        if not identifier.startswith("http")
        else domain
    )

    lead_id = upsert_lead(
        identifier=identifier,
        company_name=company_name,
        website_url=url,
        domain=domain,
    )

    # Step 1: Enrichment
    logger.info("Step 1/3 — Enriching lead: %s", identifier)
    enrichment = await enrich_lead(
        identifier, lead_id=lead_id, force_refresh=force_refresh
    )

    # Step 2: AI scoring
    logger.info("Step 2/3 — Scoring with AI (%s)", model)
    score = await score_lead_with_ai(enrichment, model=model)

    # Step 3: Persist score
    logger.info("Step 3/3 — Persisting results")
    save_score(
        lead_id=lead_id,
        budget_likelihood=score.budget_likelihood.score,
        tech_readiness=score.tech_readiness.score,
        pain_points=score.pain_points.score,
        urgency=score.urgency.score,
        overall=score.overall,
        rationale=score.to_dict(),
        model_used=model,
    )

    return LeadReport(
        identifier=identifier,
        lead_id=lead_id,
        company_name=company_name,
        website_url=url,
        domain=domain,
        enrichment=enrichment,
        score=score,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_ACTION_EMOJI = {"HOT": "🔥", "WARM": "♻️ ", "NURTURE": "🌱", "PASS": "⛔"}
_SCORE_BAR_WIDTH = 20


def _score_bar(score: float) -> str:
    filled = round(score / 10 * _SCORE_BAR_WIDTH)
    return "█" * filled + "░" * (_SCORE_BAR_WIDTH - filled)


def render_report(report: LeadReport) -> str:
    s = report.score
    action = s.recommended_action
    emoji = _ACTION_EMOJI.get(action, "")

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  CLS Corp — Lead Qualification Report                        ║",
        "╚══════════════════════════════════════════════════════════════╝",
        f"",
        f"  Company    : {report.company_name}",
        f"  Website    : {report.website_url}",
        f"  Lead ID    : {report.lead_id}",
        f"  Scored At  : {s.scored_at}",
        f"  Model      : {s.model_used}",
        f"",
        f"  ┌─────────────────────────────────────────────────────────┐",
        f"  │  ACTION: {emoji} {action:<10}   OVERALL: {s.overall:>4.1f} / 10         │",
        f"  └─────────────────────────────────────────────────────────┘",
        f"",
        f"  SCORE BREAKDOWN",
        f"  ───────────────────────────────────────────────────────────",
        f"  Budget Likelihood  {_score_bar(s.budget_likelihood.score)}  {s.budget_likelihood.score:4.1f}",
        f"  Tech Readiness     {_score_bar(s.tech_readiness.score)}  {s.tech_readiness.score:4.1f}",
        f"  Pain Points        {_score_bar(s.pain_points.score)}  {s.pain_points.score:4.1f}",
        f"  Urgency            {_score_bar(s.urgency.score)}  {s.urgency.score:4.1f}",
        f"",
        f"  EXECUTIVE SUMMARY",
        f"  ───────────────────────────────────────────────────────────",
    ]

    # Word-wrap summary
    import textwrap
    for line in textwrap.wrap(s.overall_summary, width=57):
        lines.append(f"  {line}")

    lines += [
        f"",
        f"  DIMENSION DETAIL",
        f"  ───────────────────────────────────────────────────────────",
    ]
    for dim_name, dim in [
        ("Budget Likelihood", s.budget_likelihood),
        ("Tech Readiness", s.tech_readiness),
        ("Pain Points", s.pain_points),
        ("Urgency", s.urgency),
    ]:
        lines.append(f"  [{dim.score:.1f}] {dim_name}")
        for line in textwrap.wrap(dim.reasoning, width=55):
            lines.append(f"       {line}")
        for sig in dim.signals[:4]:
            lines.append(f"       • {sig}")
        lines.append("")

    if report.enrichment.website:
        w = report.enrichment.website
        lines += [
            f"  TECH SIGNALS",
            f"  ───────────────────────────────────────────────────────────",
            f"  Stack      : {', '.join(w.tech_stack_hints) or 'None detected'}",
            f"  Industries : {', '.join(w.industries) or 'Unknown'}",
            f"  Employees  : {w.employee_count_hint or 'Not found'}",
            f"  Careers pg : {'Yes' if w.careers_page_found else 'No'}",
            f"  Pricing pg : {'Yes' if w.pricing_page_found else 'No'}",
            f"",
        ]

    lines.append(
        "══════════════════════════════════════════════════════════════"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qualifier",
        description="CLS Corp AI Lead Qualification Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              python qualifier.py stripe.com
              python qualifier.py "Acme Corp" --model deepseek/deepseek-chat
              python qualifier.py https://linear.app --output json --force
        """),
    )
    parser.add_argument("identifier", help="Company name or website URL to qualify")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        choices=["report", "json"],
        default="report",
        help="Output format (default: report)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-enrichment even if cached data exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


async def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not OPENROUTER_API_KEY:
        logger.error(
            "OPENROUTER_API_KEY environment variable is not set.\n"
            "  Export it: export OPENROUTER_API_KEY=sk-or-...\n"
            "  Or add it to a .env file in this directory."
        )
        sys.exit(1)

    try:
        report = await qualify_lead(
            identifier=args.identifier,
            model=args.model,
            force_refresh=args.force,
        )
    except Exception as exc:
        logger.error("Qualification failed: %s", exc, exc_info=args.verbose)
        sys.exit(1)

    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(render_report(report))


if __name__ == "__main__":
    asyncio.run(_main())
