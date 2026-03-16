"""
database.py — SQLite persistence layer for the Lead Qualification Agent.

Stores leads, enrichment data, qualification scores, and outreach history
with full audit trail and idempotent upserts.
"""

import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "leads.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS leads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier      TEXT    NOT NULL UNIQUE,   -- company name or canonical URL
    company_name    TEXT,
    website_url     TEXT,
    domain          TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS enrichment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id         INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    source          TEXT    NOT NULL,          -- 'website', 'linkedin', 'social', etc.
    raw_data        TEXT    NOT NULL,          -- JSON blob
    enriched_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(lead_id, source)
);

CREATE TABLE IF NOT EXISTS scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id             INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    budget_likelihood   REAL    NOT NULL CHECK(budget_likelihood BETWEEN 1 AND 10),
    tech_readiness      REAL    NOT NULL CHECK(tech_readiness BETWEEN 1 AND 10),
    pain_points         REAL    NOT NULL CHECK(pain_points BETWEEN 1 AND 10),
    urgency             REAL    NOT NULL CHECK(urgency BETWEEN 1 AND 10),
    overall             REAL    NOT NULL CHECK(overall BETWEEN 1 AND 10),
    rationale           TEXT    NOT NULL,      -- JSON with per-dimension reasoning
    model_used          TEXT    NOT NULL,
    scored_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS outreach (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id         INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    subject         TEXT    NOT NULL,
    body            TEXT    NOT NULL,
    tone            TEXT    NOT NULL DEFAULT 'professional',
    model_used      TEXT    NOT NULL,
    generated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    sent_at         TEXT,
    status          TEXT    NOT NULL DEFAULT 'draft'  -- draft | sent | bounced | replied
);

CREATE INDEX IF NOT EXISTS idx_leads_domain     ON leads(domain);
CREATE INDEX IF NOT EXISTS idx_scores_lead_id   ON scores(lead_id);
CREATE INDEX IF NOT EXISTS idx_outreach_lead_id ON outreach(lead_id);
"""


# ---------------------------------------------------------------------------
# Synchronous connection helper (sqlite3 is not async-native)
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialised at %s", db_path)


# ---------------------------------------------------------------------------
# Lead CRUD
# ---------------------------------------------------------------------------

def upsert_lead(
    identifier: str,
    company_name: Optional[str] = None,
    website_url: Optional[str] = None,
    domain: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> int:
    """Insert or update a lead record. Returns the lead id."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO leads (identifier, company_name, website_url, domain)
            VALUES (:identifier, :company_name, :website_url, :domain)
            ON CONFLICT(identifier) DO UPDATE SET
                company_name = COALESCE(:company_name, leads.company_name),
                website_url  = COALESCE(:website_url,  leads.website_url),
                domain       = COALESCE(:domain,       leads.domain),
                updated_at   = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            """,
            {
                "identifier": identifier,
                "company_name": company_name,
                "website_url": website_url,
                "domain": domain,
            },
        )
        row = conn.execute(
            "SELECT id FROM leads WHERE identifier = ?", (identifier,)
        ).fetchone()
    lead_id = row["id"]
    logger.debug("Upserted lead id=%s for identifier=%r", lead_id, identifier)
    return lead_id


def get_lead(identifier: str, db_path: Path = DB_PATH) -> Optional[dict[str, Any]]:
    """Fetch a lead by identifier, including latest score summary."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM leads WHERE identifier = ?", (identifier,)
        ).fetchone()
        if row is None:
            return None
        lead = dict(row)

        score_row = conn.execute(
            "SELECT * FROM scores WHERE lead_id = ? ORDER BY scored_at DESC LIMIT 1",
            (lead["id"],),
        ).fetchone()
        lead["latest_score"] = dict(score_row) if score_row else None

        outreach_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM outreach WHERE lead_id = ?", (lead["id"],)
        ).fetchone()["cnt"]
        lead["outreach_count"] = outreach_count

    return lead


def list_leads(db_path: Path = DB_PATH, limit: int = 50) -> list[dict[str, Any]]:
    """Return most-recently-updated leads with their latest overall score."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT l.*,
                   s.overall       AS score_overall,
                   s.scored_at     AS score_date
            FROM   leads l
            LEFT JOIN scores s ON s.id = (
                SELECT id FROM scores WHERE lead_id = l.id
                ORDER BY scored_at DESC LIMIT 1
            )
            ORDER BY l.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def save_enrichment(
    lead_id: int,
    source: str,
    data: dict[str, Any],
    db_path: Path = DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO enrichment (lead_id, source, raw_data)
            VALUES (?, ?, ?)
            ON CONFLICT(lead_id, source) DO UPDATE SET
                raw_data    = excluded.raw_data,
                enriched_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            """,
            (lead_id, source, json.dumps(data)),
        )
    logger.debug("Saved enrichment source=%r for lead_id=%s", source, lead_id)


def get_enrichment(
    lead_id: int, db_path: Path = DB_PATH
) -> dict[str, dict[str, Any]]:
    """Return all enrichment data keyed by source."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT source, raw_data FROM enrichment WHERE lead_id = ?", (lead_id,)
        ).fetchall()
    return {r["source"]: json.loads(r["raw_data"]) for r in rows}


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def save_score(
    lead_id: int,
    budget_likelihood: float,
    tech_readiness: float,
    pain_points: float,
    urgency: float,
    overall: float,
    rationale: dict[str, Any],
    model_used: str,
    db_path: Path = DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scores
                (lead_id, budget_likelihood, tech_readiness, pain_points,
                 urgency, overall, rationale, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead_id,
                budget_likelihood,
                tech_readiness,
                pain_points,
                urgency,
                overall,
                json.dumps(rationale),
                model_used,
            ),
        )
        score_id = cur.lastrowid
    logger.info(
        "Saved score id=%s for lead_id=%s  overall=%.1f", score_id, lead_id, overall
    )
    return score_id


def get_latest_score(
    lead_id: int, db_path: Path = DB_PATH
) -> Optional[dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM scores WHERE lead_id = ? ORDER BY scored_at DESC LIMIT 1",
            (lead_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["rationale"] = json.loads(result["rationale"])
    return result


# ---------------------------------------------------------------------------
# Outreach
# ---------------------------------------------------------------------------

def save_outreach(
    lead_id: int,
    subject: str,
    body: str,
    tone: str,
    model_used: str,
    db_path: Path = DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO outreach (lead_id, subject, body, tone, model_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (lead_id, subject, body, tone, model_used),
        )
        outreach_id = cur.lastrowid
    logger.info("Saved outreach id=%s for lead_id=%s", outreach_id, lead_id)
    return outreach_id


def get_outreach_history(
    lead_id: int, db_path: Path = DB_PATH
) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM outreach WHERE lead_id = ? ORDER BY generated_at DESC",
            (lead_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_outreach_sent(
    outreach_id: int,
    db_path: Path = DB_PATH,
    sent_at: Optional[str] = None,
) -> None:
    ts = sent_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE outreach SET status = 'sent', sent_at = ? WHERE id = ?",
            (ts, outreach_id),
        )
