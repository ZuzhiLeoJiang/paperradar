#!/usr/bin/env python3
"""
paperradar — personal academic literature radar.

- Fetches RSS feeds from arXiv / bioRxiv / journals
- Filters by keywords
- Ranks by semantic similarity to canonical papers
- Writes HTML digest
- Optionally posts top-N to Slack via incoming webhook

Configurable in the CONFIG section below.
"""

import os
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import feedparser
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util


# ==========================
# ======== CONFIG ==========
# ==========================

# How many days back to look
LOOKBACK_DAYS = 30

# Max items to consider per feed before filtering/ranking
MAX_ITEMS_PER_FEED = 200

# Number of top papers to include in the digest
TOP_K = 60

TODAY_TOP_K = 40
PREV_TOP_K = 30
TODAY_MIN_SCORE = 0.30

# Number of top papers to optionally post to Slack
TOP_K_SLACK = 15

# Where to save the markdown digest
OUTPUT_DIR = "./digests"

# Optional Slack incoming webhook URL (set as env var or paste string here)
SLACK_WEBHOOK_URL = os.getenv("LIT_DIGEST_SLACK_WEBHOOK", "").strip()

# ---- Feeds ----
# Add / edit as you like. URLs are RSS / Atom endpoints.

FEEDS = [
    # Preprints
    {
        “name”: “arXiv q-bio”,
        “url”: “https://export.arxiv.org/rss/q-bio”,
    },
    {
        “name”: “bioRxiv Developmental Biology”,
        “url”: “https://connect.biorxiv.org/biorxiv_xml.php?subject=developmental_biology”,
    },
    {
        “name”: “bioRxiv Cell Biology”,
        “url”: “https://connect.biorxiv.org/biorxiv_xml.php?subject=cell_biology”,
    },

    # High-impact broad journals
    {
        “name”: “Nature”,
        “url”: “http://www.nature.com/nature/current_issue/rss”,
    },
    {
        “name”: “Cell”,
        “url”: “https://www.cell.com/cell/current.rss”,
    },
    {
        “name”: “Science”,
        “url”: “http://www.sciencemag.org/rss/current.xml”,
    },
    {
        “name”: “eLife”,
        “url”: “https://elifesciences.org/rss/recent.xml”,
    },

    # Developmental biology
    {
        “name”: “Development”,
        “url”: “https://journals.biologists.com/dev/rss/current-issue.xml”,
    },
    {
        “name”: “Developmental Cell”,
        “url”: “https://www.cell.com/developmental-cell/current.rss”,
    },
    {
        “name”: “Cell Stem Cell”,
        “url”: “https://www.cell.com/cell-stem-cell/current.rss”,
    },
    {
        “name”: “Genes & Development”,
        “url”: “https://genesdev.cshlp.org/rss/current.xml”,
    },

    # Physiology & gut motility
    {
        “name”: “Journal of Physiology”,
        “url”: “https://physoc.onlinelibrary.wiley.com/feed/14697793/most-recent”,
    },
    {
        “name”: “American Journal of Physiology-GI”,
        “url”: “https://journals.physiology.org/action/showFeed?type=etoc&feed=rss&jc=ajpgi”,
    },

    # Vascular & lymphatic
    {
        “name”: “Circulation Research”,
        “url”: “https://www.ahajournals.org/action/showFeed?type=etoc&feed=rss&jc=res”,
    },
]

# ---- Keyword filters ----
# If INCLUDE_KEYWORDS is non-empty, keep items that match at least one of them in
# title or summary. Matching is case-insensitive simple substring.

INCLUDE_KEYWORDS = [
    # Gut / intestinal development
    "intestin",
    "gut development",
    "villus", "villi",
    "mesenchyme", "mesenchymal",
    "enteric",
    "perinatal", "neonatal",

    # Key cell types & markers
    "PDGFRα", "PDGFRalpha", "PDGFRA",
    "interstitial cells of Cajal", "ICC",
    "SIP syncytium",
    "smooth muscle",
    "Hand2",
    "Sox6",

    # Mechanosensing
    "Piezo",
    "mechanosensing", "mechanosensor",
    "mechanotransduction",

    # Signaling
    "Hedgehog",
    "morphogen",

    # Vascular / lymphatic
    "lacteal",
    "lymphatic",
    "PDGFRβ", "PDGFRbeta",
    "angiogenesis",
    "VEGF",

    # Single-cell & genomics
    "single-cell", "single cell",
    "scRNA-seq", "scrna",
    "spatial transcriptomics",
    "lineage tracing",
    "cell fate",
]

# If any of these appear, drop the item.
EXCLUDE_KEYWORDS = [
    "plant",
    "fungus",
    "yeast",
    "microbiome",
    "bacterial community",
    "ecology",
    "behavioral",
]

# ---- Canonical papers ----
# These seed the semantic similarity – think of them as "prototypes"
# for what you care about. Titles + short summaries is enough.

CANONICAL_PAPERS = [
    # Replace these with actual papers from your field.
    # Titles + short summaries are enough for the embedding to work well.
    {
        "title": "PDGFRα-positive cells in the intestinal wall form distinct mesenchymal populations "
                 "controlling villus morphogenesis and gut motility",
        "summary": "Single-cell transcriptomics of intestinal mesenchyme reveals two PDGFRα+ "
                   "progenitor lineages: a Sox6+ subepithelial population that drives villus "
                   "morphogenesis in response to Hedgehog signaling, and a Hand2+ population "
                   "that integrates into the SIP syncytium to regulate peristalsis."
    },
    {
        "title": "Piezo2 is the principal mechanotransduction channel for proprioception",
        "summary": "Piezo2, a mechanically activated ion channel, is required for proprioception "
                   "and light touch in mice. Conditional deletion abolishes stretch-evoked "
                   "currents in sensory neurons, establishing Piezo2 as the primary "
                   "mechanosensor in these cells."
    },
    {
        "title": "Perinatal gut mesenchyme development and SIP syncytium assembly",
        "summary": "Characterization of smooth muscle, interstitial cells of Cajal, and "
                   "PDGFRα+ cells (SIP syncytium) in the neonatal intestine. Mechanical "
                   "stimulation from feeding drives postnatal lineage specification of "
                   "mesenchymal progenitors via calcium signaling."
    },
    {
        "title": "Intestinal villus morphogenesis requires epithelial-mesenchymal crosstalk via "
                 "Hedgehog and PDGFRα signaling",
        "summary": "Conditional deletion of smoothened or PDGFRα in intestinal mesenchyme "
                   "disrupts villus clustering and subepithelial fibroblast specification, "
                   "revealing a Hedgehog-driven niche for villus formation during embryogenesis."
    },
    {
        "title": "Lacteal maintenance and dietary lipid absorption depend on VEGF-C/VEGFR3 "
                 "and PDGFRα+ stromal cells in the intestinal villus",
        "summary": "VEGF-C signaling sustains lacteal integrity in adult villi. PDGFRα+ "
                   "stromal cells wrap lacteals and support lymphatic endothelial quiescence; "
                   "their loss impairs chylomicron uptake and lipid transport."
    },
]


# Sentence-transformer model (small but decent)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



# ==========================
# ======= DATA MODEL ========
# ==========================

@dataclass
class Paper:
    title: str
    summary: str
    link: str
    published: datetime
    source: str
    score: float = math.nan  # semantic similarity score (filled later)


# ==========================
# ====== HELPERS ===========
# ==========================

def strip_html(text: str) -> str:
    """Very small HTML stripper for summaries."""
    # We keep it minimal; if you want, swap for 'beautifulsoup4'.
    import re
    return re.sub(r"<[^>]+>", "", text or "")


def parse_datetime(entry: Dict[str, Any]) -> datetime:
    """Try to get a timezone-aware datetime for the entry."""
    if "published_parsed" in entry and entry["published_parsed"]:
        return datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)
    if "updated_parsed" in entry and entry["updated_parsed"]:
        return datetime(*entry["updated_parsed"][:6], tzinfo=timezone.utc)
    # Fallback: now
    return datetime.now(timezone.utc)


def passes_keyword_filters(paper: Paper) -> bool:
    text = f"{paper.title} {paper.summary}".lower()

    if INCLUDE_KEYWORDS:
        if not any(k.lower() in text for k in INCLUDE_KEYWORDS):
            return False

    if EXCLUDE_KEYWORDS:
        if any(k.lower() in text for k in EXCLUDE_KEYWORDS):
            return False

    return True


import json
import glob
import re
from html import escape
from urllib.parse import urlsplit, urlunsplit

def canonicalize_url(url: str) -> str:
    """Normalize URLs so trivial differences don't create new IDs."""
    if not url:
        return ""
    try:
        u = urlsplit(url.strip())
        # drop fragment, normalize scheme+netloc+path; keep query (sometimes DOI/IDs live there)
        return urlunsplit((u.scheme.lower() or "https", u.netloc.lower(), u.path.rstrip("/"), u.query, ""))
    except Exception:
        return url.strip()

def paper_key(p: Paper) -> str:
    """
    Stable identity for dedup / 'seen before' comparison.
    Prefer canonicalized link; fallback to (source|title|date).
    """
    link = canonicalize_url(p.link)
    if link:
        return link
    date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return f"{p.source}|{p.title.strip().lower()}|{date_str}"

def digest_date_from_path(path: str) -> str | None:
    # expects digest_YYYY-MM-DD.html
    base = os.path.basename(path)
    m = re.match(r"digest_(\d{4}-\d{2}-\d{2})\.html$", base)
    return m.group(1) if m else None

def most_recent_non_today_digest_path(output_dir: str) -> str | None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    paths = sorted(glob.glob(os.path.join(output_dir, "digest_*.html")))
    # walk backward until we find a non-today one
    for p in reversed(paths):
        if digest_date_from_path(p) and digest_date_from_path(p) != today:
            return p
    return None

_KEYS_BLOB_RE = re.compile(r"<!--\s*DIGEST_KEYS_JSON\s*(.*?)\s*-->", re.DOTALL)

def load_papers_from_html(path: str) -> List[Paper]:
    """
    Load papers from a prior digest.

    Supports two formats:
      (A) JSON blob contains {"papers":[...]}  (new format)
      (B) JSON blob contains {"keys":[...]}    (old format) -> fallback: parse HTML cards
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        m = _KEYS_BLOB_RE.search(html)
        payload = json.loads(m.group(1)) if m else {}

        # (A) Preferred: structured papers in JSON blob
        if isinstance(payload, dict) and isinstance(payload.get("papers"), list):
            out: List[Paper] = []
            for d in payload["papers"]:
                dt = datetime.fromisoformat(d["published"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                out.append(Paper(
                    title=d.get("title", ""),
                    summary=d.get("summary", ""),
                    link=d.get("link", ""),
                    published=dt,
                    source=d.get("source", ""),
                    score=float(d["score"]) if d.get("score") is not None else math.nan,
                ))
            return [p for p in out if p.title]

        # (B) Fallback: parse the rendered HTML cards (works for your 01-04 file)
        card_re = re.compile(
            r'<div class="paper">.*?'
            r'<h3><a href="(?P<link>[^"]+)".*?>\s*(?P<title>.*?)\s*</a></h3>.*?'
            r'Source:\s*<strong>(?P<source>.*?)</strong>\s*·.*?'
            r'Date:\s*(?P<date>\d{4}-\d{2}-\d{2}).*?'
            r'Relevance score:\s*(?P<score>[\d.]+|n/a).*?'
            r'<div class="summary">(?P<summary>.*?)</div>.*?'
            r'</div>',
            re.DOTALL
        )

        out: List[Paper] = []
        for m in card_re.finditer(html):
            title = strip_html(m.group("title")).strip()
            link = strip_html(m.group("link")).strip()
            source = strip_html(m.group("source")).strip()
            date_str = strip_html(m.group("date")).strip()
            score_str = strip_html(m.group("score")).strip()
            summary = strip_html(m.group("summary")).strip()

            published = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            score = float(score_str) if score_str != "n/a" else math.nan

            if title:
                out.append(Paper(
                    title=title,
                    summary=summary,
                    link=link,
                    published=published,
                    source=source,
                    score=score,
                ))
        return out

    except Exception:
        return []


def split_new_vs_previous(papers_ranked: List[Paper], seen_keys: set[str]) -> tuple[List[Paper], List[Paper]]:
    new_items, prev_items = [], []
    for p in papers_ranked:
        k = paper_key(p)
        if k in seen_keys:
            prev_items.append(p)
        else:
            new_items.append(p)
    return new_items, prev_items


def fetch_feed(feed: Dict[str, str]) -> List[Paper]:
    print(f"Fetching feed: {feed['name']}  ({feed['url']})")
    parsed = feedparser.parse(feed["url"])
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=LOOKBACK_DAYS)

    papers: List[Paper] = []

    for entry in parsed.entries[:MAX_ITEMS_PER_FEED]:
        published = parse_datetime(entry)
        if published < cutoff:
            continue

        title = entry.get("title", "").strip()
        summary = strip_html(entry.get("summary", "")).strip()
        link = entry.get("link", "").strip() or feed["url"]

        if not title:
            continue

        paper = Paper(
            title=title,
            summary=summary,
            link=link,
            published=published,
            source=feed["name"],
        )

        if passes_keyword_filters(paper):
            papers.append(paper)

    print(f"  -> kept {len(papers)} items after filters")
    return papers


def rank_papers(papers: List[Paper]) -> List[Paper]:
    if not papers:
        return papers

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Build canonical embedding
    canon_texts = [c["title"] + ". " + c.get("summary", "") for c in CANONICAL_PAPERS]
    canon_emb = model.encode(canon_texts, convert_to_tensor=True)
    canon_emb = canon_emb.mean(dim=0, keepdim=True)  # average canonical embedding

    # Encode paper texts
    texts = [p.title + ". " + p.summary for p in papers]
    paper_emb = model.encode(texts, convert_to_tensor=True)

    sims = util.cos_sim(paper_emb, canon_emb).cpu().numpy().reshape(-1)

    for p, s in zip(papers, sims):
        p.score = float(s)

    # Sort by score descending, then by recency
    papers_sorted = sorted(
        papers,
        key=lambda p: (p.score, p.published),
        reverse=True
    )
    return papers_sorted


def format_paper_md(p: Paper) -> str:
    date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
    score_str = f"{p.score:.3f}" if not math.isnan(p.score) else "n/a"
    summary = p.summary or "_No abstract/summary available._"
    summary = textwrap.shorten(summary, width=600, placeholder="…")

    return textwrap.dedent(f"""
    ### [{p.title}]({p.link})
    - Source: **{p.source}**  
      Date: {date_str} · Relevance score: {score_str}

    {summary}
    """)


def build_markdown_digest(papers: List[Paper]) -> str:
    now = datetime.now(timezone.utc)
    header = f"# Literature Digest\n\nGenerated on {now:%Y-%m-%d %H:%M UTC}\n"
    intro = textwrap.dedent(f"""
    Time window: last {LOOKBACK_DAYS} days  
    Feeds: {', '.join(f['name'] for f in FEEDS)}  

    Ranked by semantic similarity to your canonical papers and filtered by keywords.
    """)

    body_parts = []
    for i, p in enumerate(papers, start=1):
        body_parts.append(f"---\n\n**#{i}**\n")
        body_parts.append(format_paper_md(p))

    return header + "\n" + intro + "\n\n" + "".join(body_parts)
    

def save_markdown(md: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    fname = f"digest_{now:%Y-%m-%d}.md"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved digest to {path}")
    return path

def build_html_digest(new_papers: List[Paper], prev_papers: List[Paper], history_papers: List[Paper]) -> str:
    """Return a full HTML document string with two sections:
    - Today's Feed: new since last digest
    - Previous Feed: already seen in last digest

    Also embeds a machine-readable JSON blob of paper keys as an HTML comment
    so future runs can detect duplicates.
    """
    import json
    from html import escape

    now = datetime.now(timezone.utc)

    # standalone inline CSS
    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        max-width: 900px;
        margin: 2rem auto;
        padding: 0 1rem 3rem;
        line-height: 1.5;
        color: #111827;
        background-color: #f9fafb;
    }
    h1 {
        font-size: 1.8rem;
        margin-bottom: 0.25rem;
    }
    h2 {
        font-size: 1.25rem;
        margin-top: 1.75rem;
        margin-bottom: 0.75rem;
    }
    .meta {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }
    .paper {
        background: #ffffff;
        margin: 0.75rem 0 1.0rem;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }
    .paper h3 {
        font-size: 1.05rem;
        margin: 0 0 0.25rem;
        font-weight: 650;
    }
    .paper a {
        color: #2563eb;
        text-decoration: none;
    }
    .paper a:hover {
        text-decoration: underline;
    }
    .paper .info {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .paper .summary {
        font-size: 0.95rem;
        white-space: pre-wrap;
    }
    .index {
        font-weight: 700;
        color: #4b5563;
        margin-bottom: 0.25rem;
    }
    .section-note {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    """

    header = f"""
    <h1>Literature Digest</h1>
    <div class="meta">
        Generated on {now:%Y-%m-%d %H:%M UTC}<br>
        Time window: last {LOOKBACK_DAYS} days<br>
        Feeds: {escape(", ".join(f["name"] for f in FEEDS))}
    </div>
    """

    def render_section(title: str, note: str, papers: List[Paper], start_index: int) -> str:
        if not papers:
            return f"""
            <h2>{escape(title)}</h2>
            <div class="section-note">{escape(note)}</div>
            <div class="meta">No items.</div>
            """

        blocks = [f"<h2>{escape(title)}</h2>"]
        blocks.append(f"<div class='section-note'>{escape(note)}</div>")

        for i, p in enumerate(papers, start=start_index):
            date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
            score_str = f"{p.score:.3f}" if not math.isnan(p.score) else "n/a"
            summary = p.summary or "No abstract/summary available."
            summary = textwrap.shorten(summary, width=1200, placeholder="…")

            blocks.append(f"""
            <div class="paper">
              <div class="index">#{i}</div>
              <h3><a href="{escape(p.link)}" target="_blank" rel="noopener noreferrer">
                  {escape(p.title)}
              </a></h3>
              <div class="info">
                Source: <strong>{escape(p.source)}</strong> ·
                Date: {escape(date_str)} ·
                Relevance score: {escape(score_str)}
              </div>
              <div class="summary">{escape(summary)}</div>
            </div>
            """)

        return "\n".join(blocks)

    # Build the two sections
    todays_html = render_section(
        title="Today's Feed",
        note="New since the most recent digest in the output directory.",
        papers=new_papers,
        start_index=1,
    )
    prev_html = render_section(
        title="Previous Feed",
        note="Items that also appeared in the most recent digest (overlap due to RSS windows / TOCs).",
        papers=prev_papers,
        start_index=len(new_papers) + 1,
    )

    # Embed keys so next run can read them back
    def _paper_to_dict(p: Paper) -> dict:
        return {
            "title": p.title,
            "summary": p.summary,
            "link": p.link,
            "published": p.published.astimezone(timezone.utc).isoformat(),
            "source": p.source,
            "score": p.score,
        }
    
    payload = {"papers": [_paper_to_dict(p) for p in history_papers]}

    keys_blob = f"<!-- DIGEST_KEYS_JSON {json.dumps(payload)} -->"

    body = header + todays_html + prev_html + keys_blob

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Literature Digest</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""
    return html

def save_html(html: str, path: str | None = None) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if path is None:
        now = datetime.now(timezone.utc)
        fname = f"digest_{now:%Y-%m-%d}.html"
        path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML digest to {path}")
    return path


def post_to_slack(papers: List[Paper]) -> None:
    if not SLACK_WEBHOOK_URL:
        print("No Slack webhook configured; skipping Slack notification.")
        return

    top = papers[:TOP_K_SLACK]
    if not top:
        print("No papers to post to Slack.")
        return

    lines = ["*literature digest – top hits*"]
    for p in top:
        date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
        score_str = f"{p.score:.3f}"
        line = f"• <{p.link}|{p.title}>  _(source: {p.source}, {date_str}, score {score_str})_"
        lines.append(line)

    text = "\n".join(lines)

    resp = requests.post(
        SLACK_WEBHOOK_URL,
        json={"text": text},
        timeout=10,
    )
    if resp.status_code // 100 == 2:
        print("Posted top papers to Slack.")
    else:
        print(f"Slack post failed: {resp.status_code} {resp.text}")


# ==========================
# ========= MAIN ===========
# ==========================

def main():
    all_papers: List[Paper] = []
    for feed in FEEDS:
        try:
            all_papers.extend(fetch_feed(feed))
        except Exception as e:
            print(f"Error fetching {feed['name']}: {e}")

    dedup = {}
    for p in all_papers:
        dedup.setdefault(paper_key(p), p)
    all_papers = list(dedup.values())
    
    if not all_papers:
        print("No papers found after filtering.")
        return

    # ranked = rank_papers(all_papers)
    # ranked = ranked[:TOP_K]
    
    # Load keys from most recent prior digest (if any)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_date = datetime.now(timezone.utc).date()
    
    # Path we will write to (overwrite if exists)
    today_path = os.path.join(OUTPUT_DIR, f"digest_{today}.html")
    
    # Use the most recent NON-today digest as the accumulated history baseline
    prev_path = most_recent_non_today_digest_path(OUTPUT_DIR)
    prev_papers = load_papers_from_html(prev_path) if prev_path else []
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    prev_papers = [p for p in prev_papers if p.published >= cutoff]
    seen_keys = {paper_key(p) for p in prev_papers}
    
    yesterday_date = today_date - timedelta(days=1)
    desired_dates = {today_date, yesterday_date}
    # Define "Today's feed" strictly by published date (UTC)
    today_keys = {
        paper_key(p)
        for p in all_papers
        if p.published.date() in desired_dates
    }
    
    # Merge today's RSS papers with accumulated previous papers, then dedup by key
    merged = {}
    for p in (all_papers + prev_papers):
        merged[paper_key(p)] = p
    merged_papers = list(merged.values())
    
    # Rank the merged set
    ranked = rank_papers(merged_papers)
    
    # Split AFTER ranking, but keep accumulated previous even if not in today's RSS
    new_items, prev_items = [], []
    has_history = bool(prev_papers)  # i.e., we found a prior digest
    
    for p in ranked:
        k = paper_key(p)
    
        # 1) Today's Feed: published today AND not seen before
        if (k in today_keys) and (k not in seen_keys):
            new_items.append(p)
            continue
    
        # 2) Previous Feed: anything already seen, OR (on first run) anything not from today
        if (k in seen_keys) or (not has_history and p.published.date() != today_date):
            prev_items.append(p)

    # Apply relevance threshold to Today's feed
    new_items = [p for p in new_items if (not math.isnan(p.score)) and (p.score >= TODAY_MIN_SCORE)]
    
    # Cap separately
    new_items = new_items[:TODAY_TOP_K]
    prev_items = prev_items[:PREV_TOP_K]
    
    # "history_papers" should be everything you want to remember:
    # previous history + anything you've ever shown today (today + previous candidates)
    history_dedup = {}
    for p in (prev_papers + all_papers):
        history_dedup[paper_key(p)] = p
    history_papers = list(history_dedup.values())
    
    html = build_html_digest(new_items, prev_items, history_papers)
    
    # overwrite today's file (your existing overwrite logic)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_path = os.path.join(OUTPUT_DIR, f"digest_{today}.html")
    save_html(html, path=today_path)
    
    # For Slack: usually you want only the NEW items; change if you want both.
    post_to_slack(new_items if new_items else ranked)

if __name__ == "__main__":
    main()
