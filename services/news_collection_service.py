"""
news_collection_service.py
Collects newly published news articles from GDELT (primary) and free RSS
fallback, extracts full text safely, detects duplicates, validates labels,
and stores candidates for review/approval.
"""

import logging
import threading
from typing import List, Optional

import requests

from services.bing_rss_service import BingNewsRSSProvider
from services.evidence_retrieval_service import _extract_main_text, _extract_published_date_from_html
from services.gdelt_service import GDELTSearchProvider
from services.label_validation_service import validate_candidate_label
from services.pipeline_storage import PipelineStorage, utc_now_iso
from services.search_provider import NewsSearchProvider, SearchArticle

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_QUERIES = [
    "science climate space",
    "economy inflation policy",
    "technology artificial intelligence",
    "health medical research",
]


class NewsCollectionService:
    """Periodic and manual news collector with deduplication and label validation."""

    def __init__(
        self,
        storage: PipelineStorage,
        providers: Optional[List[NewsSearchProvider]] = None,
        timeout: float = 8.0,
        session: Optional[requests.Session] = None,
    ):
        self.storage = storage
        self.providers: List[NewsSearchProvider] = providers or [
            GDELTSearchProvider(),
            BingNewsRSSProvider(),
        ]
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "VERITAS-AI-Collector/1.0 (+https://github.com/vishalwillbrite/Fake-News-Detection)"
        })
        self._run_lock = threading.Lock()

    def _fetch_raw_and_text(self, url: str) -> tuple:
        """Fetch raw HTML and extracted text for a candidate URL."""
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            if resp.status_code >= 400:
                return "", "", None
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype and not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                return "", "", None
            html = resp.text[:1_500_000]
            text = _extract_main_text(html, url=url)
            pub_date = _extract_published_date_from_html(html)
            if not text or len(text) < 80 or len(text.split()) < 12:
                return "", "", pub_date
            return html, text[:12000], pub_date
        except Exception as exc:
            logger.debug("Collection fetch failed for %s: %s", url, exc)
            return "", "", None

    def collect_news(
        self,
        queries: Optional[List[str]] = None,
        max_per_query: int = 4,
        trigger_type: str = "manual",
    ) -> dict:
        """
        Run a news collection cycle across configured queries.
        Returns summary dict of fetched, duplicates_skipped, and new_candidates.
        """
        if not self._run_lock.acquire(blocking=False):
            return {
                "success": False,
                "error": "A collection job is already running.",
            }

        started_at = utc_now_iso()
        active_queries = queries or DEFAULT_COLLECTION_QUERIES[:2]
        fetched_count = 0
        duplicates_skipped = 0
        new_candidates = 0
        errors: List[str] = []

        try:
            for q in active_queries:
                candidates: List[SearchArticle] = []
                for provider in self.providers:
                    try:
                        hits = provider.search(query=q, claim_text=q, max_results=max_per_query)
                        if hits:
                            candidates = hits
                            break
                    except Exception as exc:
                        errors.append(f"{getattr(provider, 'name', 'provider')}: {exc}")

                for art in candidates:
                    fetched_count += 1
                    raw_html, text, html_date = self._fetch_raw_and_text(art.url)
                    if not text:
                        continue

                    if self.storage.is_duplicate(art.url, text):
                        duplicates_skipped += 1
                        continue

                    val = validate_candidate_label(
                        title=art.title,
                        content=text,
                        url=art.url,
                        raw_html=raw_html,
                    )

                    cid = self.storage.add_candidate(
                        url=art.url,
                        title=art.title,
                        content=text,
                        publisher=art.publisher,
                        domain=art.domain,
                        published_date=art.published_date or html_date,
                        proposed_label=val.proposed_label,
                        status=val.status,
                        label_source=val.label_source,
                        verification_evidence=val.evidence,
                        source_url=val.source_url,
                    )
                    if cid is not None:
                        new_candidates += 1
                    else:
                        duplicates_skipped += 1

            run_status = "SUCCESS" if not errors else "PARTIAL"
            self.storage.record_collection_run(
                started_at=started_at,
                trigger_type=trigger_type,
                queries_run=len(active_queries),
                fetched_count=fetched_count,
                duplicates_skipped=duplicates_skipped,
                new_candidates_count=new_candidates,
                status=run_status,
                error_message="; ".join(errors[:3]),
            )
            logger.info(
                "Collection (%s) finished: fetched=%d, duplicates=%d, new_candidates=%d",
                trigger_type,
                fetched_count,
                duplicates_skipped,
                new_candidates,
            )
            return {
                "success": True,
                "trigger_type": trigger_type,
                "fetched_count": fetched_count,
                "duplicates_skipped": duplicates_skipped,
                "new_candidates_count": new_candidates,
                "status": run_status,
            }
        except Exception as exc:
            logger.exception("Collection run failed")
            self.storage.record_collection_run(
                started_at=started_at,
                trigger_type=trigger_type,
                queries_run=len(active_queries),
                fetched_count=fetched_count,
                duplicates_skipped=duplicates_skipped,
                new_candidates_count=new_candidates,
                status="FAILED",
                error_message=str(exc),
            )
            return {"success": False, "error": str(exc)}
        finally:
            self._run_lock.release()
