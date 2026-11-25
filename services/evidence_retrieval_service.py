"""
evidence_retrieval_service.py
Safely retrieves and extracts clean text passages from external news URLs using
requests, Trafilatura (primary), and BeautifulSoup (fallback).
Respects robots.txt, timeouts, size limits, and treats all external content as untrusted.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

from services.search_provider import SearchArticle

logger = logging.getLogger(__name__)

USER_AGENT = "VERITAS-AI-FactCheck/1.0 (+https://github.com/vishalwillbrite/Fake-News-Detection)"
MAX_CONTENT_BYTES = 1_500_000  # 1.5 MB safety cap

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "that", "this", "these", "those",
    "it", "its", "they", "their", "them", "he", "she", "his", "her",
}


@dataclass
class RetrievedEvidence:
    title: str
    url: str
    domain: str
    publisher: str
    published_date: Optional[str]
    evidence_passage: str
    accessible: bool = True
    status_note: str = "Retrieved"
    provider: str = "gdelt"

    def to_dict(self):
        return asdict(self)


def _sanitize_untrusted_text(text: str) -> str:
    """Strip control characters and normalize whitespace from untrusted webpage text."""
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_main_text(html: str, url: str = "") -> str:
    """Extract main article text using Trafilatura with BeautifulSoup fallback."""
    if not html:
        return ""

    # Primary: Trafilatura
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                html,
                url=url or None,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            if extracted and len(extracted.strip()) >= 120:
                return _sanitize_untrusted_text(extracted)
        except Exception as exc:
            logger.debug("Trafilatura extraction failed for %s: %s", url, exc)

    # Fallback: BeautifulSoup
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "svg", "nav", "footer", "header", "aside"]):
            tag.decompose()

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        joined = " ".join(p for p in paragraphs if len(p.split()) >= 5)
        if len(joined) < 100:
            joined = soup.get_text(" ", strip=True)
        return _sanitize_untrusted_text(joined)
    except Exception as exc:
        logger.debug("BeautifulSoup extraction failed for %s: %s", url, exc)
        return ""


def _extract_published_date_from_html(html: str) -> Optional[str]:
    """Best-effort extraction of publication date from common HTML meta tags."""
    if not html:
        return None
    try:
        soup = BeautifulSoup(html[:60000], "html.parser")
        meta_keys = [
            ("property", "article:published_time"),
            ("name", "pubdate"),
            ("name", "publishdate"),
            ("name", "date"),
            ("itemprop", "datePublished"),
        ]
        for attr, val in meta_keys:
            tag = soup.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                m = re.match(r"^(\d{4}-\d{2}-\d{2})", tag["content"].strip())
                if m:
                    return m.group(1)
    except Exception:
        pass
    return None


def select_relevant_passage(article_text: str, claim_text: str, max_chars: int = 450) -> str:
    """
    Select the most relevant 1-2 sentence passage from `article_text` for `claim_text`.
    """
    if not article_text:
        return ""

    claim_tokens = {
        t for t in re.findall(r"[a-z0-9]+", claim_text.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", article_text)
        if len(s.split()) >= 5
    ]
    if not sentences:
        return article_text[:max_chars].strip()

    if not claim_tokens:
        return " ".join(sentences[:2])[:max_chars].strip()

    best_score = -1.0
    best_idx = 0
    for idx, sent in enumerate(sentences):
        sent_tokens = {
            t for t in re.findall(r"[a-z0-9]+", sent.lower())
            if len(t) > 2 and t not in _STOPWORDS
        }
        if not sent_tokens:
            continue
        overlap = len(claim_tokens & sent_tokens)
        score = overlap / max(1, len(claim_tokens) ** 0.5)
        if score > best_score:
            best_score = score
            best_idx = idx

    # Take best sentence + immediate neighbor for context if it fits
    passage_parts = [sentences[best_idx]]
    if best_idx + 1 < len(sentences) and len(passage_parts[0]) + len(sentences[best_idx + 1]) + 1 <= max_chars:
        passage_parts.append(sentences[best_idx + 1])

    passage = " ".join(passage_parts).strip()
    if len(passage) > max_chars:
        passage = passage[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return passage


class EvidenceRetrievalService:
    """Fetches and extracts safe, relevant evidence passages from candidate URLs."""

    def __init__(
        self,
        timeout: float = 6.0,
        cache_ttl: int = 900,
        max_workers: int = 4,
        respect_robots: bool = True,
        session: Optional[requests.Session] = None,
    ):
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers
        self.respect_robots = respect_robots
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._page_cache: Dict[str, Tuple[float, dict]] = {}
        self._robots_cache: Dict[str, Tuple[float, Optional[RobotFileParser]]] = {}
        self._lock = threading.Lock()

    def _is_allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        try:
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            now = time.time()
            with self._lock:
                cached = self._robots_cache.get(base)
                if cached and (now - cached[0]) < 1800:
                    rp = cached[1]
                    return rp.can_fetch(USER_AGENT, url) if rp is not None else True

            robots_url = f"{base}/robots.txt"
            rp = RobotFileParser()
            resp = self.session.get(robots_url, timeout=3.0)
            if resp.status_code == 200 and resp.text:
                rp.parse(resp.text.splitlines())
                allowed = rp.can_fetch(USER_AGENT, url)
            else:
                rp = None
                allowed = True

            with self._lock:
                self._robots_cache[base] = (time.time(), rp)
            return allowed
        except Exception:
            return True

    def fetch_article_text(self, url: str) -> dict:
        """
        Fetch and extract main text from `url` with caching and safety limits.
        Returns dict: {accessible, text, published_date, status_note}.
        """
        now = time.time()
        with self._lock:
            cached = self._page_cache.get(url)
            if cached and (now - cached[0]) < self.cache_ttl:
                return dict(cached[1])

        if not url.startswith(("http://", "https://")):
            return {"accessible": False, "text": "", "published_date": None, "status_note": "Invalid URL scheme"}

        if not self._is_allowed_by_robots(url):
            result = {"accessible": False, "text": "", "published_date": None, "status_note": "Restricted by robots.txt"}
            with self._lock:
                self._page_cache[url] = (time.time(), result)
            return result

        try:
            resp = self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=True)
            if resp.status_code in (401, 403, 429):
                result = {
                    "accessible": False,
                    "text": "",
                    "published_date": None,
                    "status_note": f"Access restricted (HTTP {resp.status_code})",
                }
            elif resp.status_code >= 400:
                result = {
                    "accessible": False,
                    "text": "",
                    "published_date": None,
                    "status_note": f"Page unavailable (HTTP {resp.status_code})",
                }
            else:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if ctype and not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                    result = {
                        "accessible": False,
                        "text": "",
                        "published_date": None,
                        "status_note": f"Unsupported content type ({ctype.split(';')[0]})",
                    }
                else:
                    raw_chunks = []
                    total = 0
                    for chunk in resp.iter_content(chunk_size=16384):
                        if not chunk:
                            continue
                        raw_chunks.append(chunk)
                        total += len(chunk)
                        if total >= MAX_CONTENT_BYTES:
                            break
                    raw_bytes = b"".join(raw_chunks)
                    encoding = resp.encoding or "utf-8"
                    html = raw_bytes.decode(encoding, errors="replace")
                    extracted = _extract_main_text(html, url=url)
                    pub_date = _extract_published_date_from_html(html)
                    if extracted:
                        result = {
                            "accessible": True,
                            "text": extracted[:15000],
                            "published_date": pub_date,
                            "status_note": "Extracted article content",
                        }
                    else:
                        result = {
                            "accessible": False,
                            "text": "",
                            "published_date": pub_date,
                            "status_note": "No readable article body extracted",
                        }
        except requests.Timeout:
            result = {"accessible": False, "text": "", "published_date": None, "status_note": "Request timed out"}
        except requests.RequestException as exc:
            result = {"accessible": False, "text": "", "published_date": None, "status_note": f"Network error: {type(exc).__name__}"}

        with self._lock:
            self._page_cache[url] = (time.time(), result)
        return result

    def retrieve_evidence_for_claim(
        self,
        claim_text: str,
        articles: List[SearchArticle],
        max_sources: int = 4,
    ) -> List[RetrievedEvidence]:
        """
        Fetch content for candidate `articles` concurrently and extract relevant passages.
        """
        if not articles:
            return []

        candidates = articles[:max_sources]
        evidence_list: List[Optional[RetrievedEvidence]] = [None] * len(candidates)

        def _process(idx: int, art: SearchArticle) -> Tuple[int, RetrievedEvidence]:
            fetched = self.fetch_article_text(art.url)
            if fetched["accessible"] and fetched["text"]:
                passage = select_relevant_passage(fetched["text"], claim_text)
            else:
                # Fallback only to the real retrieved article title if body is inaccessible; never fabricate
                passage = art.title

            return idx, RetrievedEvidence(
                title=art.title,
                url=art.url,
                domain=art.domain,
                publisher=art.publisher,
                published_date=art.published_date or fetched.get("published_date"),
                evidence_passage=passage,
                accessible=fetched["accessible"],
                status_note=fetched["status_note"],
                provider=art.provider,
            )

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(candidates))) as executor:
            futures = [executor.submit(_process, i, art) for i, art in enumerate(candidates)]
            for fut in as_completed(futures):
                try:
                    idx, ev = fut.result()
                    evidence_list[idx] = ev
                except Exception as exc:
                    logger.warning("Unexpected retrieval worker error: %s", exc)

        return [ev for ev in evidence_list if ev is not None]
