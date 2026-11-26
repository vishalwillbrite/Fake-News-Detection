"""
bing_rss_service.py
Free, no-API-key Bing News RSS search provider used as a fallback when GDELT
is rate-limited (HTTP 429) or returns empty results.
Extracts direct original publisher URLs, titles, publishers, and publication dates.
"""

from email.utils import parsedate_to_datetime
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET

import requests

from services.search_provider import NewsSearchProvider, SearchArticle

logger = logging.getLogger(__name__)

BING_NEWS_RSS_ENDPOINT = "https://www.bing.com/news/search"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "have", "has", "had", "that", "this", "it", "its", "they", "their",
}


def _extract_direct_url(link: str) -> str:
    """Extract the original publisher URL from Bing News RSS redirect links."""
    if not link:
        return ""
    try:
        parsed = urlparse(link.strip())
        if "bing.com" in parsed.netloc.lower():
            qs = parse_qs(parsed.query)
            if "url" in qs and qs["url"]:
                return qs["url"][0].strip()
    except Exception:
        pass
    return link.strip()


def _format_rss_date(pub_date: Optional[str]) -> Optional[str]:
    """Parse RFC 2822 RSS pubDate into YYYY-MM-DD."""
    if not pub_date:
        return None
    try:
        dt = parsedate_to_datetime(pub_date.strip())
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return pub_date.strip()[:16] if pub_date else None


def _domain_to_publisher(domain: str) -> str:
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


def _compute_relevance(title: str, ref_text: str) -> float:
    if not title or not ref_text:
        return 0.0
    t_tokens = {t for t in re.findall(r"[a-z0-9]+", title.lower()) if len(t) > 2 and t not in _STOPWORDS}
    r_tokens = {t for t in re.findall(r"[a-z0-9]+", ref_text.lower()) if len(t) > 2 and t not in _STOPWORDS}
    if not t_tokens or not r_tokens:
        return 0.0
    overlap = len(t_tokens & r_tokens)
    return round(overlap / max(1, min(len(r_tokens), 5)), 3)


class BingNewsRSSProvider(NewsSearchProvider):
    """Free Bing News RSS search provider (no API key or billing required)."""

    name = "bing_rss"

    def __init__(
        self,
        timeout: float = 10.0,
        cache_ttl: int = 600,
        session: Optional[requests.Session] = None,
    ):
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self._cache: Dict[str, Tuple[float, List[SearchArticle]]] = {}
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    def search(self, query: str, claim_text: str = "", max_results: int = 5) -> List[SearchArticle]:
        self.last_error = None
        clean_q = " ".join(re.findall(r"[A-Za-z0-9]+", query)[:6]).strip()
        if not clean_q:
            return []

        cache_key = f"{clean_q.lower()}::{max_results}"
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (now - cached[0]) < self.cache_ttl:
                return list(cached[1])

        try:
            resp = self.session.get(
                BING_NEWS_RSS_ENDPOINT,
                params={"q": clean_q, "format": "rss"},
                timeout=self.timeout,
            )
            if resp.status_code >= 400:
                self.last_error = f"Bing RSS HTTP {resp.status_code}"
                logger.warning("Bing RSS HTTP error %s for '%s'", resp.status_code, clean_q)
                return []

            root = ET.fromstring(resp.text)
        except requests.Timeout:
            self.last_error = "Bing RSS timed out"
            logger.warning("Bing RSS request timed out for '%s'", clean_q)
            return []
        except Exception as exc:
            self.last_error = f"Bing RSS error: {type(exc).__name__}"
            logger.warning("Bing RSS error for '%s': %s", clean_q, exc)
            return []

        items = root.findall("./channel/item")
        ref_text = claim_text or clean_q
        seen_urls = set()
        seen_titles = set()
        results: List[SearchArticle] = []

        for it in items:
            title = (it.findtext("title") or "").strip()
            raw_link = (it.findtext("link") or "").strip()
            url = _extract_direct_url(raw_link)
            if not title or not url or not url.startswith(("http://", "https://")):
                continue

            norm_url = url.split("#")[0].rstrip("/").lower()
            norm_title = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
            if norm_url in seen_urls or norm_title in seen_titles:
                continue

            relevance = _compute_relevance(title, ref_text)
            if relevance < 0.12:
                continue

            domain = (urlparse(url).netloc or "").lower()
            src_el = it.find("{http://search.yahoo.com/mrss/}source") or it.find("source")
            publisher = (src_el.text.strip() if src_el is not None and src_el.text else "") or _domain_to_publisher(domain)
            pub_date = _format_rss_date(it.findtext("pubDate"))

            seen_urls.add(norm_url)
            seen_titles.add(norm_title)
            results.append(
                SearchArticle(
                    title=title,
                    url=url,
                    domain=domain,
                    publisher=publisher,
                    published_date=pub_date,
                    language="English",
                    provider=self.name,
                    relevance_score=relevance,
                )
            )

        results.sort(key=lambda a: a.relevance_score, reverse=True)
        final_results = results[:max_results]

        with self._lock:
            self._cache[cache_key] = (time.time(), list(final_results))

        logger.info("Bing RSS returned %d relevant articles for '%s'", len(final_results), clean_q)
        return final_results
