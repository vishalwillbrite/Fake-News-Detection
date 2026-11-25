"""
gdelt_service.py
Free GDELT DOC 2.0 API integration with caching, rate limiting, deduplication,
relevance filtering, and resilient error handling.
"""

import logging
import re
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests

from services.search_provider import NewsSearchProvider, SearchArticle

logger = logging.getLogger(__name__)

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "have", "has", "had", "that", "this", "it", "its", "they", "their",
}


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication (strip fragment, trailing slash, tracking params)."""
    try:
        parsed = urlparse(url.strip())
        path = parsed.path.rstrip("/") or "/"
        # Drop query params like utm_* for dedupe comparison
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", "", ""))
    except Exception:
        return url.strip().lower()


def _format_gdelt_date(seendate: Optional[str]) -> Optional[str]:
    """Convert GDELT seendate (e.g., 20260512T143000Z) to YYYY-MM-DD."""
    if not seendate:
        return None
    m = re.match(r"^(\d{4})(\d{2})(\d{2})", str(seendate).strip())
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return str(seendate)


def _domain_to_publisher(domain: str) -> str:
    """Convert a domain like 'www.reuters.com' into a readable publisher name."""
    if not domain:
        return "Unknown Publisher"
    host = domain.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 2:
        name = parts[0]
        return name.upper() if len(name) <= 4 else name.capitalize()
    return host


def _sanitize_gdelt_query(query: str) -> str:
    """Strip characters that break GDELT DOC query syntax."""
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    # GDELT requires words; keep tokens >= 2 chars
    cleaned = [t for t in tokens if len(t) >= 2]
    return " ".join(cleaned[:8])


def _compute_title_relevance(title: str, claim_or_query: str) -> float:
    """Compute lexical overlap ratio between candidate article title and claim terms."""
    if not title or not claim_or_query:
        return 0.0
    title_tokens = {
        t for t in re.findall(r"[a-z0-9]+", title.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }
    query_tokens = {
        t for t in re.findall(r"[a-z0-9]+", claim_or_query.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }
    if not title_tokens or not query_tokens:
        return 0.0
    overlap = len(title_tokens & query_tokens)
    return round(overlap / max(1, min(len(query_tokens), 6)), 3)


class GDELTSearchProvider(NewsSearchProvider):
    """GDELT DOC 2.0 API news search provider."""

    name = "gdelt"

    def __init__(
        self,
        timeout: float = 15.0,
        cache_ttl: int = 600,
        min_request_interval: float = 5.1,
        session: Optional[requests.Session] = None,
    ):
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.min_request_interval = min_request_interval
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "VERITAS-AI-FactCheck/1.0 (+https://github.com/vishalwillbrite/Fake-News-Detection)"
        })
        self._cache: Dict[str, Tuple[float, List[SearchArticle]]] = {}
        self._lock = threading.Lock()
        self._last_request_ts = 0.0

    def _respect_rate_limit(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_ts
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self._last_request_ts = time.time()

    def search(self, query: str, claim_text: str = "", max_results: int = 5) -> List[SearchArticle]:
        """
        Query GDELT DOC 2.0 API and return deduplicated, relevance-filtered SearchArticle objects.
        Never raises on network/API errors; returns empty list and logs warnings.
        """
        sanitized_query = _sanitize_gdelt_query(query)
        if not sanitized_query:
            return []

        cache_key = f"{sanitized_query.lower()}::{max_results}"
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (now - cached[0]) < self.cache_ttl:
                return list(cached[1])

        params = {
            "query": f"{sanitized_query} sourcelang:english",
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max(15, max_results * 3),
            "sort": "HybridRel",
        }

        data = None
        for attempt in range(2):
            self._respect_rate_limit()
            try:
                resp = self.session.get(
                    GDELT_DOC_ENDPOINT,
                    params=params,
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    logger.warning("GDELT rate limited (429); waiting 5.2s to respect API rate limit.")
                    time.sleep(5.2)
                    continue
                if resp.status_code >= 400:
                    logger.warning("GDELT HTTP error %s for query '%s'", resp.status_code, sanitized_query)
                    return []

                # GDELT occasionally returns plain-text warnings with HTTP 200
                text_body = (resp.text or "").strip()
                if not text_body or not text_body.startswith("{"):
                    logger.warning("GDELT returned non-JSON response for '%s': %s", sanitized_query, text_body[:120])
                    return []

                data = resp.json()
                break
            except requests.Timeout:
                logger.warning("GDELT request timed out for query '%s' (attempt %d)", sanitized_query, attempt + 1)
            except requests.RequestException as exc:
                logger.warning("GDELT request error for query '%s': %s", sanitized_query, exc)
                return []
            except ValueError as exc:
                logger.warning("GDELT JSON parse error for query '%s': %s", sanitized_query, exc)
                return []

        if not isinstance(data, dict):
            return []

        raw_articles = data.get("articles") or []
        if not isinstance(raw_articles, list):
            return []

        seen_urls = set()
        seen_titles = set()
        results: List[SearchArticle] = []
        ref_text = claim_text or sanitized_query

        for item in raw_articles:
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            if not url or not title or not url.startswith(("http://", "https://")):
                continue

            # Filter non-English if GDELT metadata specifies language
            lang = (item.get("language") or "").strip()
            if lang and lang.lower() not in ("english", "eng", "en"):
                continue

            norm_url = _normalize_url(url)
            norm_title = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
            if norm_url in seen_urls or norm_title in seen_titles:
                continue

            relevance = _compute_title_relevance(title, ref_text)
            if relevance < 0.15:
                continue

            domain = (item.get("domain") or urlparse(url).netloc or "").lower()
            publisher = _domain_to_publisher(domain)
            pub_date = _format_gdelt_date(item.get("seendate"))

            seen_urls.add(norm_url)
            seen_titles.add(norm_title)
            results.append(
                SearchArticle(
                    title=title,
                    url=url,
                    domain=domain,
                    publisher=publisher,
                    published_date=pub_date,
                    language=lang or "English",
                    provider=self.name,
                    relevance_score=relevance,
                )
            )

        results.sort(key=lambda a: a.relevance_score, reverse=True)
        final_results = results[:max_results]

        with self._lock:
            self._cache[cache_key] = (time.time(), list(final_results))

        logger.info("GDELT returned %d relevant unique articles for query '%s'", len(final_results), sanitized_query)
        return final_results
