"""
outreach.py — Personalised Cold Outreach Email Generator (CLS Corp).

Generates highly contextualised cold email drafts using:
  • Lead qualification scores from qualifier.py
  • Enrichment data from enricher.py (tech stack, industry, pain signals)
  • Configurable tone: professional | casual | executive | technical
  • Multiple template variants via temperature sampling

Usage:
  python outreach.py stripe.com
  python outreach.py "Acme Corp" --tone casual --variants 3
  python outreach.py stripe.com --output json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from database import (
    get_enrichment,
    get_latest_score,
    get_outreach_history,
    init_db,
    save_outreach,
    upsert_lead,
)

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("OUTREACH_MODEL", "deepseek/deepseek-chat")
SENDER_NAME = os.getenv("SENDER_NAME", "Alex Rivera")
SENDER_TITLE = os.getenv("SENDER_TITLE", "Head of AI Partnerships")
SENDER_COMPANY = os.getenv("SENDER_COMPANY", "CLS Corp")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "alex@clscorp.ai")
SITE_URL = os.getenv("SITE_URL", "https://clscorp.ai")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmailDraft:
    subject: str
    body: str
    tone: str
    variant: int = 1
    model_used: str = DEFAULT_MODEL
    word_count: int = 0

    def __post_init__(self) -> None:
        if self.word_count == 0:
            self.word_count = len(self.body.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "body": self.body,
            "tone": self.tone,
            "variant": self.variant,
            "model_used": self.model_used,
            "word_count": self.word_count,
        }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_TONE_GUIDES = {
    "professional": dedent("""
        Tone: Professional and consultative.
        - Use "I" sparingly; lead with insight about THEIR business.
        - No buzzwords (synergy, leverage, etc.).
        - Clear value proposition in the first 2 sentences.
        - One soft CTA (15-min call, not "buy now").
        - 120-160 words max.
    """).strip(),

    "casual": dedent("""
        Tone: Friendly and direct, as if from a peer who noticed something interesting.
        - First-name basis, conversational.
        - Short sentences. One idea per paragraph.
        - Light humour is OK if it fits naturally.
        - CTA feels low-pressure: "happy to share what we found if useful."
        - 100-140 words max.
    """).strip(),

    "executive": dedent("""
        Tone: Concise, ROI-first. Written for a C-suite reader with 30 seconds to spare.
        - Lead with a specific business outcome (cost, revenue, speed).
        - No fluff. No bullet lists. Tight paragraphs.
        - Name a specific number or benchmark if possible.
        - CTA: request for a 10-min executive briefing.
        - 90-120 words max.
    """).strip(),

    "technical": dedent("""
        Tone: Peer-to-peer technical. Written as if from one engineer/CTO to another.
        - Reference specific tech stack items found in enrichment.
        - Show you understand their architecture.
        - Mention a concrete technical approach (e.g., "a lightweight LLM layer on top of your existing Postgres pipeline").
        - CTA: link to a technical case study or demo repo.
        - 130-170 words max.
    """).strip(),
}

_SYSTEM_PROMPT = dedent("""
    You are an elite B2B sales copywriter at CLS Corp, an AI automation agency.
    You write cold emails that get replies — not because they're aggressive, but
    because they are hyper-relevant and demonstrate genuine research.

    You MUST respond with a JSON object containing exactly two keys:
      "subject": the email subject line (max 60 chars, no click-bait)
      "body":    the full email body (plain text, newlines as \\n)

    No markdown fences. No extra keys. No explanation outside the JSON.

    Rules:
    - Never lie or exaggerate — use only the provided context.
    - Never use placeholder text like [INSERT X HERE].
    - Personalise using the specific data points given (tech stack, industry, scores).
    - The opening line must NOT start with "I" or "My name is".
    - Always sign off with the sender details provided.
""").strip()


def _build_outreach_prompt(
    identifier: str,
    enrichment_data: dict[str, Any],
    score_data: Optional[dict[str, Any]],
    tone: str,
    variant_hint: str = "",
) -> str:
    tone_guide = _TONE_GUIDES.get(tone, _TONE_GUIDES["professional"])

    website = enrichment_data.get("website", {})
    social = enrichment_data.get("social", {})

    tech_stack = website.get("tech_stack_hints", [])
    industries = website.get("industries", [])
    title = website.get("title", identifier)
    description = website.get("description", "")
    has_careers = website.get("careers_page_found", False)
    has_pricing = website.get("pricing_page_found", False)
    employee_hint = website.get("employee_count_hint", "")
    raw_snippet = website.get("raw_text_snippet", "")[:1500]
    github_repos = social.get("github_public_repos", 0)
    github_stars = social.get("github_stars_total", 0)
    github_org = social.get("github_org", "")

    score_context = ""
    if score_data:
        rationale = score_data.get("rationale", {})
        pain_signals = []
        if isinstance(rationale.get("pain_points"), dict):
            pain_signals = rationale["pain_points"].get("signals", [])
        urgency_signals = []
        if isinstance(rationale.get("urgency"), dict):
            urgency_signals = rationale["urgency"].get("signals", [])

        score_context = (
            f"\nQualification Scores:\n"
            f"  Overall          : {score_data.get('overall', 'N/A')}/10\n"
            f"  Budget Likelihood: {score_data.get('budget_likelihood', 'N/A')}/10\n"
            f"  Pain Points      : {score_data.get('pain_points', 'N/A')}/10\n"
            f"  Urgency          : {score_data.get('urgency', 'N/A')}/10\n"
            f"  Key Pain Signals : {'; '.join(pain_signals[:3]) or 'see snippet'}\n"
            f"  Urgency Signals  : {'; '.join(urgency_signals[:3]) or 'see snippet'}\n"
        )

    prompt = (
        f"## Lead Context\n"
        f"Company / URL      : {identifier}\n"
        f"Website Title      : {title}\n"
        f"Meta Description   : {description[:200]}\n"
        f"Industries Detected: {', '.join(industries) or 'unknown'}\n"
        f"Tech Stack         : {', '.join(tech_stack) or 'unknown'}\n"
        f"Employee Hint      : {employee_hint or 'unknown'}\n"
        f"Has Careers Page   : {has_careers}\n"
        f"Has Pricing Page   : {has_pricing}\n"
        f"GitHub Org         : {github_org or 'N/A'}  "
        f"({github_repos} repos, {github_stars} stars)\n"
        f"{score_context}\n"
        f"## Website Snippet\n{raw_snippet}\n\n"
        f"## Sender Info\n"
        f"Name    : {SENDER_NAME}\n"
        f"Title   : {SENDER_TITLE}\n"
        f"Company : {SENDER_COMPANY}\n"
        f"Email   : {SENDER_EMAIL}\n"
        f"Website : {SITE_URL}\n\n"
        f"## Tone & Length Guide\n{tone_guide}\n"
    )

    if variant_hint:
        prompt += f"\n## Variant Instruction\n{variant_hint}\n"

    return prompt


_VARIANT_HINTS = [
    "",  # variant 1: no extra constraint
    "This is variant 2: open with a specific observation about their tech stack or GitHub presence.",
    "This is variant 3: open with an industry pain-point or market trend relevant to their sector, then tie it to CLS Corp's solution.",
    "This is variant 4: open with a bold ROI claim backed by a benchmark (you may invent a plausible statistic for illustration).",
]


# ---------------------------------------------------------------------------
# AI call
# ---------------------------------------------------------------------------

async def _call_openrouter(
    system: str,
    user: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": SITE_URL,
                "X-Title": f"{SENDER_COMPANY} Outreach",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": 800,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()

    raw: str = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

async def generate_outreach(
    identifier: str,
    lead_id: Optional[int] = None,
    tone: str = "professional",
    variants: int = 1,
    model: str = DEFAULT_MODEL,
) -> list[EmailDraft]:
    """
    Generate one or more personalised cold email drafts for a lead.

    Args:
        identifier: Company name or URL (must already exist in DB).
        lead_id:    Database lead id; resolved from identifier if None.
        tone:       One of professional | casual | executive | technical.
        variants:   Number of alternative drafts to generate (1-4).
        model:      OpenRouter model ID.

    Returns:
        List of EmailDraft objects.
    """
    init_db()

    if tone not in _TONE_GUIDES:
        raise ValueError(f"Invalid tone '{tone}'. Choose from: {list(_TONE_GUIDES)}")

    variants = max(1, min(variants, 4))

    # Resolve lead_id
    if lead_id is None:
        from enricher import _normalise_url
        from urllib.parse import urlparse

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

    enrichment_data = get_enrichment(lead_id)
    if not enrichment_data:
        logger.warning(
            "No enrichment found for lead_id=%s. Outreach will be less personalised.",
            lead_id,
        )

    score_data = get_latest_score(lead_id)
    if not score_data:
        logger.warning("No score found for lead_id=%s. Proceeding without scores.", lead_id)

    logger.info(
        "Generating %d outreach variant(s) for %s  tone=%s  model=%s",
        variants,
        identifier,
        tone,
        model,
    )

    # Build all variant tasks
    async def _generate_variant(variant_index: int) -> EmailDraft:
        hint = _VARIANT_HINTS[variant_index] if variant_index < len(_VARIANT_HINTS) else ""
        user_prompt = _build_outreach_prompt(
            identifier=identifier,
            enrichment_data=enrichment_data,
            score_data=score_data,
            tone=tone,
            variant_hint=hint,
        )
        # Slight temperature increase for later variants → more creative
        temperature = 0.4 + variant_index * 0.15

        parsed = await _call_openrouter(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            model=model,
            temperature=temperature,
        )

        draft = EmailDraft(
            subject=parsed["subject"],
            body=parsed["body"],
            tone=tone,
            variant=variant_index + 1,
            model_used=model,
        )

        # Persist to DB
        save_outreach(
            lead_id=lead_id,
            subject=draft.subject,
            body=draft.body,
            tone=tone,
            model_used=model,
        )
        logger.info("Generated variant %d  (%d words)", draft.variant, draft.word_count)
        return draft

    # Run all variants concurrently
    tasks = [_generate_variant(i) for i in range(variants)]
    drafts: list[EmailDraft] = await asyncio.gather(*tasks)
    return drafts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_drafts(drafts: list[EmailDraft], identifier: str) -> str:
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  CLS Corp — Outreach Email Drafts                            ║",
        "╚══════════════════════════════════════════════════════════════╝",
        f"  Lead     : {identifier}",
        f"  Tone     : {drafts[0].tone if drafts else 'N/A'}",
        f"  Variants : {len(drafts)}",
        f"  Model    : {drafts[0].model_used if drafts else 'N/A'}",
        "",
    ]

    for draft in drafts:
        lines += [
            f"  ──────────────────────────── VARIANT {draft.variant} ──────────────────",
            f"  Subject : {draft.subject}",
            f"  Words   : {draft.word_count}",
            "",
        ]
        for para in draft.body.split("\n"):
            lines.append(f"  {para}")
        lines.append("")

    lines.append("══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="outreach",
        description="CLS Corp AI Outreach Email Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              python outreach.py stripe.com
              python outreach.py "Acme Corp" --tone executive --variants 2
              python outreach.py stripe.com --tone technical --output json
        """),
    )
    parser.add_argument("identifier", help="Company name or URL (must have been qualified first)")
    parser.add_argument(
        "--tone",
        choices=list(_TONE_GUIDES.keys()),
        default="professional",
        help="Email tone (default: professional)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        choices=range(1, 5),
        metavar="N",
        help="Number of variants to generate 1-4 (default: 1)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


async def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    try:
        drafts = await generate_outreach(
            identifier=args.identifier,
            tone=args.tone,
            variants=args.variants,
            model=args.model,
        )
    except Exception as exc:
        logger.error("Outreach generation failed: %s", exc, exc_info=args.verbose)
        sys.exit(1)

    if args.output == "json":
        print(json.dumps([d.to_dict() for d in drafts], indent=2))
    else:
        print(render_drafts(drafts, args.identifier))


if __name__ == "__main__":
    asyncio.run(_main())
