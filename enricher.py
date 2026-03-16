"""
enricher.py — Lead enrichment pipeline for the Lead Qualification Agent.

Aggregates data from multiple sources:
  • Company website scraping  (BeautifulSoup / httpx)
  • LinkedIn public profile   (stub — real impl requires Proxycurl or SalesNav API)
  • Social media presence      (Twitter/X, GitHub, Crunchbase lightweight checks)

All enrichers are async and run concurrently via asyncio.gather.
Results are persisted to SQLite via database.py.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from database import get_enrichment, save_enrichment, upsert_lead

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP client factory
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CLSCorpLeadBot/1.0; +https://clscorp.ai/bot)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=HEADERS,
        timeout=TIMEOUT,
        follow_redirects=True,
        http2=True,
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WebsiteData:
    url: str
    title: str = ""
    description: str = ""
    tech_stack_hints: list[str] = field(default_factory=list)
    social_links: dict[str, str] = field(default_factory=dict)
    employee_count_hint: str = ""
    industries: list[str] = field(default_factory=list)
    careers_page_found: bool = False
    blog_found: bool = False
    pricing_page_found: bool = False
    contact_email: str = ""
    raw_text_snippet: str = ""
    fetch_duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class LinkedInData:
    profile_url: str = ""
    company_size: str = ""
    industry: str = ""
    headquarters: str = ""
    founded_year: str = ""
    specialties: list[str] = field(default_factory=list)
    follower_count: str = ""
    recent_posts_count: int = 0
    is_stub: bool = True
    note: str = "LinkedIn enrichment requires Proxycurl or Sales Navigator API key"


@dataclass
class SocialPresenceData:
    twitter_handle: str = ""
    twitter_followers: str = ""
    github_org: str = ""
    github_public_repos: int = 0
    github_stars_total: int = 0
    crunchbase_slug: str = ""
    crunchbase_total_funding: str = ""
    g2_rating: str = ""
    glassdoor_rating: str = ""
    error: Optional[str] = None


@dataclass
class EnrichmentResult:
    lead_id: int
    identifier: str
    website: Optional[WebsiteData] = None
    linkedin: Optional[LinkedInData] = None
    social: Optional[SocialPresenceData] = None
    enriched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        def _asdict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _asdict(v) for k, v in obj.__dict__.items()}
            return obj

        return {
            "lead_id": self.lead_id,
            "identifier": self.identifier,
            "website": _asdict(self.website) if self.website else None,
            "linkedin": _asdict(self.linkedin) if self.linkedin else None,
            "social": _asdict(self.social) if self.social else None,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_TECH_PATTERNS: dict[str, list[str]] = {
    "React": ["react", "reactjs", "__REACT_"],
    "Next.js": ["_next/", "next.js"],
    "Vue": ["vue.js", "vuejs", "__vue__"],
    "Angular": ["ng-version", "angular"],
    "WordPress": ["wp-content", "wp-json", "WordPress"],
    "Shopify": ["cdn.shopify.com", "Shopify.theme"],
    "Salesforce": ["force.com", "salesforce", "pardot"],
    "HubSpot": ["hs-scripts", "hubspot", "hbspt"],
    "Intercom": ["intercom", "widget.intercom.io"],
    "Stripe": ["js.stripe.com"],
    "Segment": ["cdn.segment.com", "analytics.js"],
    "Google Analytics": ["gtag(", "google-analytics.com", "ga("],
    "Hotjar": ["hotjar"],
    "Cloudflare": ["__cf_bm", "cloudflare"],
    "AWS": ["amazonaws.com", "cloudfront.net"],
    "Vercel": ["vercel.app", "_vercel"],
    "Kubernetes": ["kubernetes", "k8s"],
}

_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "SaaS / Software": ["saas", "software", "platform", "api", "developer tools"],
    "E-commerce": ["shop", "store", "cart", "checkout", "product catalog"],
    "FinTech": ["fintech", "payment", "banking", "finance", "lending", "wallet"],
    "HealthTech": ["health", "medical", "clinical", "hipaa", "patient", "ehr"],
    "EdTech": ["education", "learning", "course", "lms", "student", "curriculum"],
    "MarTech": ["marketing", "crm", "leads", "campaigns", "seo", "ads"],
    "Logistics": ["logistics", "supply chain", "fleet", "shipping", "freight"],
    "Real Estate": ["real estate", "property", "mortgage", "realty", "mls"],
    "Manufacturing": ["manufacturing", "industrial", "factory", "production"],
    "Consulting": ["consulting", "advisory", "strategy", "professional services"],
}

_EMPLOYEE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(\d{1,5})\+?\s*employees\b", re.I), "explicit"),
    (re.compile(r"\bteam of\s+(\d+)\b", re.I), "team"),
    (re.compile(r"\b(\d{3,5})\s+people\b", re.I), "people"),
]


def _detect_tech_stack(html: str) -> list[str]:
    found = []
    for tech, patterns in _TECH_PATTERNS.items():
        if any(p.lower() in html.lower() for p in patterns):
            found.append(tech)
    return found


def _detect_industries(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        industry
        for industry, keywords in _INDUSTRY_KEYWORDS.items()
        if any(kw in text_lower for kw in keywords)
    ]


def _extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else ""


def _detect_employee_count(text: str) -> str:
    for pattern, _ in _EMPLOYEE_HINTS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return ""


def _normalise_url(url_or_name: str) -> str:
    """Best-effort: turn a bare company name into a guessed URL, or normalise existing URL."""
    s = url_or_name.strip()
    if re.match(r"^https?://", s):
        return s
    # Looks like a domain?
    if re.match(r"^[\w\-]+\.[a-z]{2,}$", s, re.I):
        return f"https://{s}"
    # Treat as company name → attempt .com
    slug = re.sub(r"[^a-z0-9]", "", s.lower())
    return f"https://www.{slug}.com"


# ---------------------------------------------------------------------------
# Website enricher
# ---------------------------------------------------------------------------

async def enrich_website(url: str, client: httpx.AsyncClient) -> WebsiteData:
    data = WebsiteData(url=url)
    t0 = time.monotonic()
    try:
        response = await client.get(url)
        response.raise_for_status()
        html = response.text
        data.fetch_duration_ms = round((time.monotonic() - t0) * 1000, 1)

        soup = BeautifulSoup(html, "html.parser")

        # Title & meta description
        data.title = (soup.title.string or "").strip() if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
        if meta_desc and meta_desc.get("content"):
            data.description = meta_desc["content"].strip()[:500]

        # Raw text snippet for AI context
        body_text = soup.get_text(separator=" ", strip=True)
        data.raw_text_snippet = re.sub(r"\s{2,}", " ", body_text)[:3000]

        # Tech stack detection (HTML source)
        data.tech_stack_hints = _detect_tech_stack(html)

        # Industry detection
        data.industries = _detect_industries(body_text)

        # Employee count hints
        data.employee_count_hint = _detect_employee_count(body_text)

        # Social links
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            for platform, pattern in [
                ("twitter", r"twitter\.com/(?!share|intent)"),
                ("linkedin", r"linkedin\.com/company/"),
                ("github", r"github\.com/(?![^/]+/issues|topics)"),
                ("youtube", r"youtube\.com/(?:c/|@|channel/)"),
                ("facebook", r"facebook\.com/"),
            ]:
                if re.search(pattern, href, re.I) and platform not in data.social_links:
                    data.social_links[platform] = (
                        href if href.startswith("http") else urljoin(base, href)
                    )

        # Contact email
        data.contact_email = _extract_email(body_text)

        # Internal page presence
        all_links = {
            a["href"].lower()
            for a in soup.find_all("a", href=True)
            if isinstance(a["href"], str)
        }
        data.careers_page_found = any(
            kw in lnk for lnk in all_links for kw in ("career", "job", "hiring", "join-us")
        )
        data.blog_found = any(kw in lnk for lnk in all_links for kw in ("blog", "news", "insights"))
        data.pricing_page_found = any(kw in lnk for lnk in all_links for kw in ("pricing", "plans", "cost"))

        logger.info(
            "Website enriched: %s  [%d ms]  tech=%s",
            url,
            data.fetch_duration_ms,
            data.tech_stack_hints,
        )
    except httpx.HTTPStatusError as exc:
        data.error = f"HTTP {exc.response.status_code}"
        logger.warning("Website fetch failed for %s: %s", url, data.error)
    except Exception as exc:
        data.error = str(exc)
        logger.warning("Website enrichment error for %s: %s", url, exc)

    return data


# ---------------------------------------------------------------------------
# LinkedIn enricher (stub)
# ---------------------------------------------------------------------------

async def enrich_linkedin(company_name: str, domain: str) -> LinkedInData:
    """
    Production implementation would call:
      POST https://nubela.co/proxycurl/api/linkedin/company
    with an API key.  For this demo we return a clearly-labelled stub.
    """
    await asyncio.sleep(0)  # preserve async contract
    slug = re.sub(r"[^a-z0-9\-]", "-", company_name.lower()).strip("-")
    return LinkedInData(
        profile_url=f"https://www.linkedin.com/company/{slug}",
        is_stub=True,
        note=(
            "Stub data — integrate Proxycurl (nubela.co) or LinkedIn Sales Navigator "
            "API to retrieve real company_size, headcount, recent posts, etc."
        ),
    )


# ---------------------------------------------------------------------------
# Social / public data enricher
# ---------------------------------------------------------------------------

async def enrich_social(
    company_name: str,
    domain: str,
    website_data: Optional[WebsiteData],
    client: httpx.AsyncClient,
) -> SocialPresenceData:
    data = SocialPresenceData()

    # Extract handles from website social links if available
    if website_data and website_data.social_links:
        for platform, url in website_data.social_links.items():
            if platform == "twitter":
                match = re.search(r"twitter\.com/(@?[\w]+)", url)
                if match:
                    data.twitter_handle = match.group(1).lstrip("@")
            if platform == "github":
                match = re.search(r"github\.com/([\w\-]+)", url)
                if match:
                    data.github_org = match.group(1)

    # GitHub public repo stats (unauthenticated, rate-limited to 60 req/hr)
    if data.github_org:
        try:
            resp = await client.get(
                f"https://api.github.com/orgs/{data.github_org}/repos",
                params={"per_page": 100, "type": "public"},
                headers={**HEADERS, "Accept": "application/vnd.github+json"},
            )
            if resp.status_code == 200:
                repos = resp.json()
                data.github_public_repos = len(repos)
                data.github_stars_total = sum(r.get("stargazers_count", 0) for r in repos)
                logger.info(
                    "GitHub: org=%s  repos=%d  stars=%d",
                    data.github_org,
                    data.github_public_repos,
                    data.github_stars_total,
                )
        except Exception as exc:
            logger.debug("GitHub enrichment error: %s", exc)

    return data


# ---------------------------------------------------------------------------
# Main enrichment orchestrator
# ---------------------------------------------------------------------------

async def enrich_lead(
    identifier: str,
    lead_id: Optional[int] = None,
    *,
    force_refresh: bool = False,
) -> EnrichmentResult:
    """
    Run all enrichers concurrently for a given identifier (URL or company name).

    Args:
        identifier:     Company name or website URL.
        lead_id:        Pre-existing database ID; will upsert if None.
        force_refresh:  Re-enrich even if cached data exists.

    Returns:
        EnrichmentResult with populated sub-objects.
    """
    url = _normalise_url(identifier)
    parsed = urlparse(url)
    domain = parsed.netloc.lstrip("www.")
    company_name = domain.split(".")[0].title() if not identifier.startswith("http") else domain

    # Ensure lead record exists
    if lead_id is None:
        lead_id = upsert_lead(
            identifier=identifier,
            company_name=company_name,
            website_url=url,
            domain=domain,
        )

    # Check cache
    if not force_refresh:
        existing = get_enrichment(lead_id)
        if existing:
            logger.info("Using cached enrichment for lead_id=%s", lead_id)
            # Reconstruct lightweight result from cache
            result = EnrichmentResult(lead_id=lead_id, identifier=identifier)
            return result

    logger.info("Starting enrichment pipeline for: %s  (id=%s)", identifier, lead_id)

    async with _make_client() as client:
        website_task = asyncio.create_task(enrich_website(url, client))
        # LinkedIn stub doesn't need the client
        linkedin_future = enrich_linkedin(company_name, domain)

        website_data = await website_task
        linkedin_data, social_data = await asyncio.gather(
            linkedin_future,
            enrich_social(company_name, domain, website_data, client),
        )

    result = EnrichmentResult(
        lead_id=lead_id,
        identifier=identifier,
        website=website_data,
        linkedin=linkedin_data,
        social=social_data,
    )

    # Persist each enrichment source
    save_enrichment(lead_id, "website", website_data.__dict__)
    save_enrichment(lead_id, "linkedin", linkedin_data.__dict__)
    save_enrichment(lead_id, "social", social_data.__dict__)

    logger.info("Enrichment complete for lead_id=%s", lead_id)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from database import init_db

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Enrich a lead from public data sources.")
    parser.add_argument("identifier", help="Company name or website URL")
    parser.add_argument("--force", action="store_true", help="Force re-enrichment")
    parser.add_argument("--json", action="store_true", dest="output_json", help="Output raw JSON")
    args = parser.parse_args()

    init_db()
    result = asyncio.run(enrich_lead(args.identifier, force_refresh=args.force))

    if args.output_json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        w = result.website
        if w:
            print(f"\n=== Website Enrichment: {w.url} ===")
            print(f"  Title        : {w.title}")
            print(f"  Description  : {w.description[:120]}...")
            print(f"  Tech Stack   : {', '.join(w.tech_stack_hints) or 'None detected'}")
            print(f"  Industries   : {', '.join(w.industries) or 'Unknown'}")
            print(f"  Employees    : {w.employee_count_hint or 'Not found'}")
            print(f"  Social Links : {w.social_links}")
            print(f"  Has Careers  : {w.careers_page_found}")
            print(f"  Has Pricing  : {w.pricing_page_found}")
            print(f"  Contact Email: {w.contact_email or 'Not found'}")
        li = result.linkedin
        if li:
            print(f"\n=== LinkedIn (Stub) ===")
            print(f"  Profile URL  : {li.profile_url}")
            print(f"  Note         : {li.note}")
        s = result.social
        if s:
            print(f"\n=== Social Presence ===")
            print(f"  GitHub       : {s.github_org or 'Not found'}  ({s.github_public_repos} repos, {s.github_stars_total} stars)")
            print(f"  Twitter      : {s.twitter_handle or 'Not found'}")
