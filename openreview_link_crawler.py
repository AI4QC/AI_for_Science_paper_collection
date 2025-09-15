#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
openreview_link_crawler.py

High-level purpose:
    Given a CSV containing at least a column named 'title' (paper title) and
    an 'virtualsite_url' (a NeurIPS / conference virtual site entry URL),
    attempt to resolve each paper to its OpenReview forum URL.

Core workflow per row:
    1. (Optional) Use an override mapping if provided (override file) to directly map titles -> forum link.
    2. If an existing valid openreview_link already present (and not resuming unmatched), keep it.
    3. Try to fetch the virtual site page(s) with multiple URL variants; parse out the forum link if present.
    4. If enabled and still unresolved, perform fallback web search by constructing progressively shorter
       search queries from the normalized title. Parse search results HTML to extract candidate forum IDs,
       score them with fuzzy matching (rapidfuzz), and accept if score >= --fuzzy-high
       or mark ambiguous if fuzzy-low <= score < fuzzy-high.
    5. Collect statistics and optionally write diagnostic JSON for ambiguous/unmatched rows.
    6. Optionally drop internal meta columns before writing the final output CSV
       unless --keep-meta is specified.

Important notes:
    * Title column name MUST be lower-case 'title' (NOT 'Title'). The script enforces this.
    * Parallelization is provided via ThreadPoolExecutor.
    * A small local JSON cache (websearch_cache_v4.json) stores raw HTML of past queries to reduce
      repeated network calls.
    * Normalization aggressively strips punctuation, LaTeX math, Greek letters (converted to names),
      superscripts, and non-alphanumerics for robust fuzzy matching.

Example usage (basic):
    python openreview_link_crawler.py \
        --input my_neurips2024.csv \
        --enable-web-search-fallback \
        --max-workers 5

Resume unmatched rows only:
    python openreview_link_crawler.py \
        --input my_neurips2024.csv \
        --enable-web-search-fallback \
        --resume-unmatched

Use override links:
    python openreview_link_crawler.py \
        --input my_neurips2024.csv \
        --override-file override_links.csv

Keep meta columns (match_status, match_strategy, etc.) in final output:
    --keep-meta

Manual Post-step:
    "You need to delete the last column by yourself :)" (original author's note)
"""

import argparse
import os
import re
import sys
import json
import time
import random
import unicodedata
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
from rapidfuzz import fuzz
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- Constants ---------------- #
# URL / pattern constants used to discover forum links and interpret search results.
FORUM_PREFIX = "https://openreview.net/forum?id="
OPENREVIEW_SEARCH_BASE = "https://openreview.net/search?term="

# Regexes to capture forum URLs or forum IDs embedded in HTML/JS
OPENREVIEW_FULL_REGEX = re.compile(r"https://openreview\.net/forum\?id=[A-Za-z0-9_-]+")
OPENREVIEW_RELAX_REGEX = re.compile(r"forum\?id=[A-Za-z0-9_-]+")
FORUM_ID_ONLY_REGEX = re.compile(r'["\']?forum["\']?\s*[:=]\s*["\']([A-Za-z0-9_-]{6,})["\']')
POSSIBLE_JSON_REGEX = re.compile(r'\{.*\}', re.DOTALL)  # (Unused currently, placeholder for future JSON extraction)

# Stopwords to reduce noise in search query terms.
STOPWORDS = set("""
a an the of for to in on and with without by via from at is are be been being this that those these using use toward towards
into over under between among across against as it its their his her our your toward towarding towarded data model learning
""".split())

# Mapping of Greek letters and superscripts to plain text for normalization.
GREEK_MAP = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon', 'θ': 'theta',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau',
    'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega', '∇': 'nabla'
}
SUPERSCRIPT_MAP = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}


# -------------- Normalization -------------- #
def normalize_title(title: str) -> str:
    """
    Normalize a paper title into a simplified lower-case token string.

    Steps:
        1. Unicode NFKC normalization
        2. Replace Greek letters with English names
        3. Replace superscript digits
        4. Remove LaTeX math segments $...$ and \( \)
        5. Convert to ASCII with unidecode
        6. Lower-case
        7. Flatten remaining punctuation to spaces, keep alphanumerics
        8. Collapse multiple spaces

    Args:
        title: Original title (may be None / not a string)

    Returns:
        A normalized string; empty string if input not valid.
    """
    if not isinstance(title, str):
        return ""
    t = unicodedata.normalize("NFKC", title)

    # Replace Greek and superscript characters
    for k, v in GREEK_MAP.items():
        t = t.replace(k, v)
    for k, v in SUPERSCRIPT_MAP.items():
        t = t.replace(k, v)

    # Remove LaTeX math (simplistic)
    t = re.sub(r'\$[^$]*\$', ' ', t)
    t = re.sub(r'\\\(|\\\)', ' ', t)

    # ASCII fold and lower
    t = unidecode(t).lower()

    # Remove caret-superscripts like ^2
    t = re.sub(r'\^(\d+)', r'\1', t)

    # Keep only alphanumerics + whitespace
    t = re.sub(r'[^a-z0-9\s]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def build_search_tokens(title: str, max_tokens: int) -> List[str]:
    """
    Build a list of tokens for queries by normalizing and dropping stopwords.

    Args:
        title: Paper title
        max_tokens: Maximum number of tokens to keep

    Returns:
        Filtered token list (non-stopwords first), truncated to max_tokens.
        Falls back to all tokens if filtering removes everything.
    """
    base = normalize_title(title).split()
    tokens = [w for w in base if w not in STOPWORDS and len(w) > 2]
    if not tokens:
        tokens = base
    return tokens[:max_tokens]


def build_queries_multi(title: str, max_tokens: int, min_tokens: int = 2) -> List[str]:
    """
    Construct a descending list of queries of length N, N-1, ... >= min_tokens.

    For example, tokens: ['adaptive','graph','neural']
    => ["adaptive graph neural", "adaptive graph", "adaptive"]

    Args:
        title: Original title
        max_tokens: Upper bound of tokens considered
        min_tokens: Minimum number of tokens a query must contain

    Returns:
        List of query strings (no duplicates, descending token count).
    """
    tokens = build_search_tokens(title, max_tokens)
    queries = []
    for k in range(len(tokens), min_tokens - 1, -1):
        q = " ".join(tokens[:k])
        if q and q not in queries:
            queries.append(q)
    return queries


# -------------- HTTP -------------- #
def fetch(url: str, timeout: int, retries: int, base_sleep: float, jitter: float,
          accept=(200,), giveup=(400, 401, 403, 404)) -> Optional[str]:
    """
    Perform an HTTP GET with retry/backoff.

    Args:
        url: Target URL
        timeout: Per-attempt timeout (seconds)
        retries: Maximum number of attempts
        base_sleep: Base delay (exponential factor base_sleep * 1.6^(attempt-1))
        jitter: Random additive jitter upper bound
        accept: Status codes considered success
        giveup: Status codes that cause immediate permanent failure

    Returns:
        Response text on success, None on failure or give-up status.
    """
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ORFetcherV4/1.0)"})
    attempt = 0
    while attempt < retries:
        try:
            r = sess.get(url, timeout=timeout)
            if r.status_code in accept:
                return r.text
            if r.status_code in giveup:
                # Do not continue retrying for permanent categories
                return None
        except Exception:
            # Swallow exceptions and retry
            pass
        attempt += 1
        time.sleep(base_sleep * (1.6 ** (attempt - 1)) + random.uniform(0, jitter))
    return None


# -------------- Virtual Parsing -------------- #
def normalize_virtual_url(u: str) -> str:
    """
    Normalize a 'virtual site' URL string (fix missing schemes, duplicate slashes).

    Args:
        u: Raw URL (may be partial)

    Returns:
        Normalized absolute HTTPS URL or empty string if input invalid.
    """
    if not u:
        return ""
    u = u.strip()
    if u.startswith("//"):
        u = "https:" + u
    elif not u.startswith("http"):
        # Add https:// if scheme absent
        u = "https://" + u.lstrip("/")
    # Collapse accidental double slashes (except protocol part)
    u = re.sub(r'(?<!:)//+', '/', u)
    # Fix single slash protocol e.g. https:/something
    if u.startswith("https:/") and not u.startswith("https://"):
        u = u.replace("https:/", "https://", 1)
    return u


def extract_forum_link_deep(html: str) -> str:
    """
    Attempt multiple strategies to discover an OpenReview forum link in provided HTML.

    Strategies:
        1. Direct full forum URL match via regex.
        2. Relative 'forum?id=...' match.
        3. Parsing all anchor tags for href or data-forum-id.
        4. Searching generic attributes of all tags.
        5. Inspecting <script> blocks for embedded forum ids or partial strings.

    Args:
        html: Raw HTML text

    Returns:
        Forum URL if found, else empty string.
    """
    if not html:
        return ""
    # Direct full match
    m = OPENREVIEW_FULL_REGEX.search(html)
    if m:
        return m.group(0)

    # Relaxed match
    m2 = OPENREVIEW_RELAX_REGEX.search(html)
    if m2:
        fid = m2.group(0).split("forum?id=")[-1]
        return FORUM_PREFIX + fid

    # Parse HTML
    soup = BeautifulSoup(html, "html.parser")

    # Walk anchor tags
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith(FORUM_PREFIX):
            return h
        if "forum?id=" in h:
            fid = h.split("forum?id=")[-1]
            return FORUM_PREFIX + fid
        fid_attr = a.get("data-forum-id")
        if fid_attr:
            return FORUM_PREFIX + fid_attr

    # Look in all element attributes
    for tag in soup.find_all(attrs=True):
        for k, v in tag.attrs.items():
            if isinstance(v, str) and "forum?id=" in v:
                fid = v.split("forum?id=")[-1]
                return FORUM_PREFIX + fid

    # Inspect scripts for embedded references
    for sc in soup.find_all("script"):
        txt = sc.string or sc.get_text()
        if not txt:
            continue
        # IDs followed by colon or assignment, or references to forum?id=
        for m3 in FORUM_ID_ONLY_REGEX.finditer(txt):
            fid = m3.group(1)
            if len(fid) >= 6:
                return FORUM_PREFIX + fid
        if "forum?id=" in txt:
            m4 = OPENREVIEW_RELAX_REGEX.search(txt)
            if m4:
                fid = m4.group(0).split("forum?id=")[-1]
                return FORUM_PREFIX + fid
    return ""


def try_virtual_all(base_url: str, timeout: int, retries: int,
                    base_sleep: float, jitter: float,
                    save_fail_html: bool, idx: int) -> Tuple[str, str]:
    """
    Attempt multiple 'variant' forms of a virtual site URL (e.g., adding query params
    or trailing slash) to maximize chance of containing the forum link.

    For each variant:
        * Fetch
        * If HTML present, try extraction
        * Optionally persist failing HTML for debugging

    Args:
        base_url: Raw virtual site URL from CSV
        timeout, retries, base_sleep, jitter: HTTP parameters (see fetch)
        save_fail_html: If True, store HTML / markers for debugging
        idx: Row index (for naming debug files)

    Returns:
        (forum_link, reason_label)
        forum_link = empty string if not found
        reason_label = indicates which variant found it or failure category
    """
    variants = []
    core = normalize_virtual_url(base_url)
    if not core:
        return "", "virtual_no_url"

    # Collect plausible permutations
    variants.append((core, "virtual_primary"))
    variants.append((core + "?showControls=true", "virtual_variant_controls"))
    variants.append((core + "?showPaper=true", "virtual_variant_paper"))
    if not core.endswith("/"):
        variants.append((core + "/", "virtual_variant_slash"))

    for url, label in variants:
        html = fetch(url, timeout, retries, base_sleep, jitter)
        if html:
            link = extract_forum_link_deep(html)
            if link:
                return link, label
            else:
                # Save HTML for diagnostics if extraction failed
                if save_fail_html:
                    os.makedirs("debug_html/virtual", exist_ok=True)
                    with open(f"debug_html/virtual/row_{idx}_{label}.html", "w", encoding="utf-8") as f:
                        f.write(html[:200000])
        else:
            if save_fail_html:
                os.makedirs("debug_html/virtual_fail", exist_ok=True)
                with open(f"debug_html/virtual_fail/row_{idx}_{label}.txt", "w", encoding="utf-8") as f:
                    f.write(f"FETCH_FAIL {url}")
    return "", "virtual_all_failed"


# -------------- Web Search Fallback -------------- #
def parse_search_html(html: str) -> List[Tuple[str, str]]:
    """
    Extract candidate (title, forum_link) pairs from an OpenReview search results page.

    Process:
        * Parse anchor tags referencing '/forum?id='
        * Inspect script blocks for 'forum?id=' occurrences or object fields 'forum'
        * Attempt to pair discovered forum IDs with embedded '"title":' strings

    Args:
        html: Search results page HTML

    Returns:
        List of (title_text, forum_link). Title may be empty if not retrieved.
    """
    anchors = []
    if not html:
        return anchors
    soup = BeautifulSoup(html, "html.parser")

    # Anchor tags with forum links directly in href
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if "/forum?id=" in h:
            fid = h.split("forum?id=")[-1].split("&")[0]
            anchors.append((a.get_text(" ", strip=True), FORUM_PREFIX + fid))

    # Collect script text for deeper patterns
    script_texts = []
    for sc in soup.find_all("script"):
        txt = sc.string or sc.get_text()
        if not txt:
            continue
        if "forum?id=" in txt or '"forum"' in txt:
            script_texts.append(txt)

    # Find forum IDs
    fids = set()
    for txt in script_texts:
        for m in re.finditer(r'forum\?id=([A-Za-z0-9_-]{6,})', txt):
            fids.add(m.group(1))
        for m in FORUM_ID_ONLY_REGEX.finditer(txt):
            fids.add(m.group(1))

    # Attempt to extract titles from script JSON-ish data
    titles = []
    for txt in script_texts:
        for m in re.finditer(r'"title"\s*:\s*"([^"]{5,200})"', txt):
            raw = m.group(1)
            # Unescape unicode sequences
            if "\\u" in raw:
                try:
                    raw = bytes(raw, 'utf-8').decode('unicode_escape')
                except Exception:
                    pass
            titles.append(raw)

    # Pair fids with titles if lengths match; otherwise output minimal data
    if fids and titles:
        if len(fids) == len(titles):
            for t, fid in zip(titles, list(fids)):
                anchors.append((t, FORUM_PREFIX + fid))
        else:
            # Mismatch: at least record links
            for fid in fids:
                anchors.append(("", FORUM_PREFIX + fid))
    elif fids:
        for fid in fids:
            anchors.append(("", FORUM_PREFIX + fid))
    return anchors


def web_search_multi_queries(title: str,
                             high: int, low: int,
                             max_tokens: int,
                             timeout: int, retries: int,
                             base_sleep: float, jitter: float,
                             save_fail_html: bool, idx: int,
                             cache: Dict[str, str], cache_dirty: List[bool]) -> Tuple[str, str, float, List[dict], str]:
    """
    Execute fallback web search by generating multiple queries (descending token count)
    until enough candidates (or one high-confidence) is found.

    Steps:
        1. Build queries
        2. For each query:
            * Use cached HTML if present; else fetch and cache
            * Parse candidate links
            * Accumulate until 50 raw candidates or queries exhausted
        3. Deduplicate links; compute fuzzy scores vs normalized target
        4. Choose best candidate:
              score >= high        => accepted
              low <= score < high  => ambiguous
              else                 => unmatched

    Args:
        title: Paper title
        high: Fuzzy acceptance threshold
        low: Fuzzy ambiguous threshold
        max_tokens: Max tokens for initial query construction
        timeout, retries, base_sleep, jitter: HTTP fetch parameters
        save_fail_html: Save failing HTML markers for debugging
        idx: Row index (for file naming)
        cache: Query -> raw HTML cache
        cache_dirty: Single-item list used as mutable boolean flag

    Returns:
        (status, link, score, scored_candidates, reason_code)
        status: 'accepted' | 'ambiguous' | 'unmatched'
        link: forum link if accepted, else empty
        score: best fuzzy score
        scored_candidates: list of candidate dicts [{candidate_title, query, score, link, normalized}, ...]
        reason_code: textual reason (e.g., 'web_search_accept', 'web_search_no_candidates')
    """
    queries = build_queries_multi(title, max_tokens, min_tokens=2)
    all_candidates = []

    for q in queries:
        if q in cache:
            html = cache[q]
        else:
            html = fetch(OPENREVIEW_SEARCH_BASE + requests.utils.quote(q),
                         timeout, retries, base_sleep, jitter)
            if html:
                cache[q] = html
                cache_dirty[0] = True
            else:
                if save_fail_html:
                    os.makedirs("debug_html/search_fail", exist_ok=True)
                    with open(f"debug_html/search_fail/row_{idx}_query_{len(q.split())}.txt",
                              "w", encoding="utf-8") as f:
                        f.write(f"HTTP_FAIL query={q}")
                continue

        pairs = parse_search_html(html)
        for (t, link) in pairs:
            all_candidates.append((t, link, q))
        # Stop early if too many raw candidates (prevents runaway)
        if len(all_candidates) >= 50:
            break

    dedup = {}
    for t, link, q in all_candidates:
        if link not in dedup:
            dedup[link] = (t, q)

    scored = []
    norm_target = normalize_title(title)
    for link, (t, q) in dedup.items():
        norm_c = normalize_title(t)
        score = fuzz.token_set_ratio(norm_target, norm_c) if norm_c else 0
        scored.append({
            "candidate_title": t,
            "query": q,
            "score": score,
            "link": link,
            "normalized": norm_c
        })
    scored.sort(key=lambda x: x["score"], reverse=True)

    if not scored:
        return "unmatched", "", 0.0, scored, "web_search_no_candidates"

    best = scored[0]
    if best["score"] >= high:
        return "accepted", best["link"], best["score"], scored, "web_search_accept"
    elif best["score"] >= low:
        return "ambiguous", "", best["score"], scored, "web_search_ambiguous"
    return "unmatched", "", best["score"], scored, "web_search_unmatched"


# -------------- Override -------------- #
def load_override(path: Optional[str]) -> Dict[str, str]:
    """
    Load an override CSV that maps normalized titles to forum links.

    Expected columns (case-insensitive alternatives):
        - title or Title
        - openreview_link or link

    Only links starting with FORUM_PREFIX are accepted.

    Args:
        path: Path to override CSV

    Returns:
        Dict: normalized_title -> forum_link
    """
    if not path or not os.path.exists(path):
        return {}
    import csv
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = r.get("title") or r.get("Title")
            link = r.get("openreview_link") or r.get("link")
            if t and link and link.startswith(FORUM_PREFIX):
                mp[normalize_title(t)] = link.strip()
    print(f"[OVERRIDE] loaded {len(mp)} entries.")
    return mp


# -------------- Row Processing -------------- #
def process_row(idx: int, row: pd.Series, args, override_map,
                web_cache, web_cache_dirty, salvage: bool = False):
    """
    Process a single DataFrame row:
        - Apply salvage logic if resuming unmatched (skip rows already decided)
        - Apply override map
        - Keep existing openreview_link if valid (and not salvage mode)
        - Attempt virtual site probing
        - Fallback to web search if enabled
        - Return structured result controlling how the DataFrame is updated

    Args:
        idx: Row index
        row: Row (Series) with required fields (at least 'title')
        args: Parsed CLI args
        override_map: Preloaded override dictionary
        web_cache: Query->HTML cache
        web_cache_dirty: Flag container for marking cache modifications
        salvage: Whether we are resuming only unmatched rows

    Returns:
        Dict with keys:
            row_index, openreview_link, match_status, match_strategy,
            match_score, reason_detail, (optional) candidates
    """
    title = str(row["title"])
    norm = normalize_title(title)

    existing_link = row.get("openreview_link", "")
    existing_status = row.get("match_status", "")
    existing_reason = row.get("reason_detail", "")

    # Salvage mode: keep previously resolved statuses (anything not unmatched)
    if salvage:
        if existing_status != "unmatched":
            return {
                "row_index": idx,
                "openreview_link": existing_link,
                "match_status": existing_status,
                "match_strategy": row.get("match_strategy", ""),
                "match_score": row.get("match_score", ""),
                "reason_detail": existing_reason or "resume_skip"
            }

    # Override check
    if norm in override_map:
        return {
            "row_index": idx,
            "openreview_link": override_map[norm],
            "match_status": "accepted",
            "match_strategy": "override",
            "match_score": "",
            "reason_detail": "override"
        }

    # Already has a valid forum link (non-salvage full run)
    if not salvage and isinstance(existing_link, str) and existing_link.startswith(FORUM_PREFIX):
        return {
            "row_index": idx,
            "openreview_link": existing_link,
            "match_status": existing_status or "accepted",
            "match_strategy": row.get("match_strategy", "previous"),
            "match_score": row.get("match_score", ""),
            "reason_detail": existing_reason or "previous"
        }

    # Attempt virtual site extraction unless skipped
    virtual_reason = ""
    if not args.skip_virtual:
        vurl = str(row.get("virtualsite_url", "")).strip()
        if vurl:
            link, vr = try_virtual_all(
                vurl,
                timeout=args.virtual_timeout,
                retries=args.virtual_retries,
                base_sleep=args.retry_base_sleep,
                jitter=args.retry_jitter,
                save_fail_html=args.save_fail_html,
                idx=idx
            )
            if link:
                return {
                    "row_index": idx,
                    "openreview_link": link,
                    "match_status": "accepted",
                    "match_strategy": "virtualsite",
                    "match_score": "",
                    "reason_detail": vr
                }
            else:
                virtual_reason = vr
        else:
            virtual_reason = "virtual_no_url"

    # Web search fallback
    if args.enable_web_search_fallback:
        status, link, score, candidates, wr = web_search_multi_queries(
            title,
            args.fuzzy_high,
            args.fuzzy_low,
            args.web_query_tokens,
            timeout=args.web_search_timeout,
            retries=args.web_search_retries,
            base_sleep=args.retry_base_sleep,
            jitter=args.retry_jitter,
            save_fail_html=args.save_fail_html,
            idx=idx,
            cache=web_cache,
            cache_dirty=web_cache_dirty
        )
        if status == "accepted":
            return {
                "row_index": idx,
                "openreview_link": link,
                "match_status": "accepted",
                "match_strategy": "websearch",
                "match_score": score,
                "reason_detail": f"{virtual_reason}+{wr}" if virtual_reason else wr
            }
        elif status == "ambiguous":
            return {
                "row_index": idx,
                "openreview_link": "",
                "match_status": "ambiguous",
                "match_strategy": "websearch",
                "match_score": score,
                "reason_detail": f"{virtual_reason}+{wr}" if virtual_reason else wr,
                "candidates": candidates
            }
        else:
            combined = f"{virtual_reason}+{wr}" if virtual_reason else wr
            return {
                "row_index": idx,
                "openreview_link": "",
                "match_status": "unmatched",
                "match_strategy": "none",
                "match_score": "",
                "reason_detail": combined
            }

    # If no fallback or all failed
    return {
        "row_index": idx,
        "openreview_link": "",
        "match_status": "unmatched",
        "match_strategy": "none",
        "match_score": "",
        "reason_detail": virtual_reason or "unmatched"
    }


# -------------- Main -------------- #
def autodetect_conf_year(name: str):
    """
    Best-effort extraction of conference name and year from filename.

    Pattern:
        (NeurIPS|ICLR|ICML|CVPR|ECCV|AAAI|KDD)[-_]?(\d{4})

    Args:
        name: File path or name

    Returns:
        (conference, year) fallback to ("Conference", "Year") if not found.
    """
    base = os.path.basename(name)
    m = re.search(r"(NeurIPS|ICLR|ICML|CVPR|ECCV|AAAI|KDD)[-_]?(\d{4})", base, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    return "Conference", "Year"


def main():
    """
    CLI orchestration:
        * Parse arguments
        * Read input CSV
        * Initialize meta columns
        * Load overrides and web search cache
        * Determine row set (all vs unmatched resume)
        * Dispatch row processing (serial or threaded)
        * Collect results, compute statistics
        * Write ambiguous / unmatched diagnostics
        * Persist updated cache & output CSV
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV file containing at least a 'title' column.")
    ap.add_argument("--output", help="Output CSV path (default: <Conference>_<Year>_openreview_link.csv)")
    ap.add_argument("--conference", help="Override inferred conference label.")
    ap.add_argument("--year", help="Override inferred year.")
    ap.add_argument("--override-file", help="CSV file with manual mappings (title, openreview_link).")
    ap.add_argument("--skip-virtual", action="store_true", help="Skip virtual site probing and go straight to search fallback.")
    ap.add_argument("--enable-web-search-fallback", action="store_true", help="Enable fallback OpenReview site search.")
    ap.add_argument("--web-search-timeout", type=int, default=20)
    ap.add_argument("--web-search-retries", type=int, default=2)
    ap.add_argument("--web-query-tokens", type=int, default=6, help="Max tokens for initial search queries.")
    ap.add_argument("--fuzzy-high", type=int, default=96, help="Score >= this => accept.")
    ap.add_argument("--fuzzy-low", type=int, default=90, help="Score >= fuzzy_low and < fuzzy_high => ambiguous.")
    ap.add_argument("--virtual-timeout", type=int, default=18)
    ap.add_argument("--virtual-retries", type=int, default=3)
    ap.add_argument("--retry-base-sleep", type=float, default=1.0)
    ap.add_argument("--retry-jitter", type=float, default=0.3)
    ap.add_argument("--row-sleep", type=float, default=0.4, help="Sleep between rows (serial mode) to reduce server load.")
    ap.add_argument("--row-jitter", type=float, default=0.3)
    ap.add_argument("--max-workers", type=int, default=4, help=">1 uses threaded processing.")
    ap.add_argument("--limit-rows", type=int, help="Process only first N rows.")
    ap.add_argument("--resume-unmatched", action="store_true", help="Only reprocess previously unmatched rows.")
    ap.add_argument("--save-fail-html", action="store_true", help="Persist failing HTML snippets for debugging.")
    ap.add_argument("--keep-meta", action="store_true",
                    help="Keep meta columns (match_status, match_strategy, etc.) in output.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print("[ERROR] input not found")
        sys.exit(1)

    conf, year = autodetect_conf_year(args.input)
    if args.conference:
        conf = args.conference
    if args.year:
        year = args.year
    if not args.output:
        args.output = f"{conf}_{year}_openreview_link.csv"

    df = pd.read_csv(args.input)
    if "title" not in df.columns:
        print("[ERROR] need title column (exactly 'title')")
        sys.exit(1)
    if args.limit_rows:
        df = df.head(args.limit_rows).copy()

    # Ensure meta columns exist (some may be absent on first run).
    for col in ["openreview_link", "match_status", "match_strategy", "match_score", "reason_detail", "status_log"]:
        if col not in df.columns:
            df[col] = ""

    override_map = load_override(args.override_file)

    # Load / initialize web search cache
    web_cache_path = "websearch_cache_v4.json"
    if os.path.exists(web_cache_path):
        try:
            web_cache = json.load(open(web_cache_path, "r", encoding="utf-8"))
        except Exception:
            web_cache = {}
    else:
        web_cache = {}
    web_cache_dirty = [False]  # Mutable flag so functions can mark modifications

    # Pick row indices (all or unmatched)
    if args.resume_unmatched:
        indices = [i for i in df.index if df.at[i, "match_status"] == "unmatched"]
        print(f"[RESUME] unmatched rows: {len(indices)} / {len(df)}")
    else:
        indices = list(df.index)
        print(f"[RUN] total rows to process: {len(indices)}")

    results = {}

    def dispatch(i):
        return process_row(i, df.loc[i], args, override_map, web_cache, web_cache_dirty,
                           salvage=args.resume_unmatched)

    # Serial or threaded processing
    if args.max_workers <= 1:
        for i in tqdm(indices, desc="Processing"):
            results[i] = dispatch(i)
            time.sleep(args.row_sleep + random.uniform(0, args.row_jitter))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            fut_map = {ex.submit(dispatch, i): i for i in indices}
            for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Processing"):
                res = fut.result()
                results[res["row_index"]] = res

    ambiguous = []
    unmatched = []

    # Apply results to DataFrame
    for i in df.index:
        if i in results:
            r = results[i]
            df.at[i, "openreview_link"] = r["openreview_link"]
            df.at[i, "match_status"] = r["match_status"]
            df.at[i, "match_strategy"] = r["match_strategy"]
            df.at[i, "match_score"] = r["match_score"]
            df.at[i, "reason_detail"] = r["reason_detail"]
            if r["match_status"] == "ambiguous":
                ambiguous.append({
                    "row_index": i,
                    "title": df.at[i, "title"],
                    "candidates": r.get("candidates", [])
                })
            if r["match_status"] == "unmatched":
                unmatched.append({
                    "row_index": i,
                    "title": df.at[i, "title"],
                    "normalized": normalize_title(df.at[i, "title"]),
                    "reason_detail": r["reason_detail"]
                })

    # Stats before optional column drop
    total = len(df)
    accepted = (df["match_status"] == "accepted").sum()
    amb_cnt = (df["match_status"] == "ambiguous").sum()
    un_cnt = (df["match_status"] == "unmatched").sum()
    print("========== STATS (before strip) ==========")
    print(f"Total: {total}")
    print(f"Accepted: {accepted}")
    print(f"Ambiguous: {amb_cnt}")
    print(f"Unmatched: {un_cnt}")
    print("Strategy distribution:")
    print(df["match_strategy"].value_counts())
    print("Reason detail (top 15):")
    print(df["reason_detail"].value_counts().head(15))
    print("================================")

    # Diagnostics
    if ambiguous:
        json.dump(ambiguous, open("ambiguous.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[INFO] ambiguous: {len(ambiguous)}")
    if unmatched:
        json.dump(unmatched, open("unmatched.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[INFO] unmatched: {len(unmatched)}")
        pd.DataFrame(unmatched).to_csv("unmatched_diagnosis.csv", index=False)

    # Persist web cache if updated
    if web_cache_dirty[0]:
        json.dump(web_cache, open(web_cache_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("[CACHE] websearch cache updated.")

    # Conditional meta column removal
    if args.keep_meta:
        df_out = df
    else:
        drop_cols = [c for c in ["match_status", "match_strategy", "match_score", "status_log"]
                     if c in df.columns]
        if drop_cols:
            print(f"[OUTPUT] dropping meta columns: {drop_cols}")
            df_out = df.drop(columns=drop_cols)
        else:
            df_out = df

    # Write final CSV
    df_out.to_csv(args.output, index=False)
    print(f"[DONE] wrote: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")