"""
verification_orchestrator.py
Coordinates the end-to-end external news verification pipeline and manages
background verification jobs for non-blocking Flask execution.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

from services.claim_extraction_service import ExtractedClaim, extract_claims
from services.evidence_retrieval_service import EvidenceRetrievalService, RetrievedEvidence
from services.evidence_verification_service import (
    ClaimVerificationResult,
    EvidenceVerificationService,
)
from services.final_decision_service import FinalAssessment, compute_final_assessment
from services.gdelt_service import GDELTSearchProvider
from services.search_provider import NewsSearchProvider, SearchArticle

logger = logging.getLogger(__name__)


class VerificationOrchestrator:
    """
    Runs claim extraction -> multi-provider search -> evidence retrieval ->
    local NLI verification -> final combined assessment.
    Supports both synchronous execution and background job polling.
    """

    def __init__(
        self,
        providers: Optional[List[NewsSearchProvider]] = None,
        retrieval_service: Optional[EvidenceRetrievalService] = None,
        verification_service: Optional[EvidenceVerificationService] = None,
        cache_ttl: int = 600,
    ):
        self.providers: List[NewsSearchProvider] = providers or [GDELTSearchProvider()]
        self.retrieval_service = retrieval_service or EvidenceRetrievalService()
        self.verification_service = verification_service or EvidenceVerificationService()
        self.cache_ttl = cache_ttl

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="veritas-verify")
        self._jobs: Dict[str, dict] = {}
        self._result_cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def _search_providers(self, query: str, claim_text: str, max_results: int = 4) -> List[SearchArticle]:
        """Query all configured search providers and merge deduplicated results."""
        merged: List[SearchArticle] = []
        seen_urls = set()
        for provider in self.providers:
            try:
                hits = provider.search(query=query, claim_text=claim_text, max_results=max_results)
                for art in hits:
                    norm = art.url.strip().lower().rstrip("/")
                    if norm not in seen_urls:
                        seen_urls.add(norm)
                        merged.append(art)
            except Exception as exc:
                logger.warning("Search provider '%s' failed: %s", getattr(provider, "name", "unknown"), exc)
        return merged[:max_results]

    def verify_article(
        self,
        news_text: str,
        ml_result: dict,
        progress_cb: Optional[Callable[[str, int], None]] = None,
        max_claims: int = 3,
    ) -> dict:
        """
        Execute the full external verification pipeline synchronously and return
        a JSON-serializable dictionary combining ML and external verification.
        """
        start_ts = time.time()
        text_hash = hashlib.sha256(news_text.strip().encode("utf-8")).hexdigest()

        now = time.time()
        with self._lock:
            cached = self._result_cache.get(text_hash)
            if cached and (now - cached[0]) < self.cache_ttl:
                if progress_cb:
                    progress_cb("Loaded cached external verification results.", 100)
                cached_payload = dict(cached[1])
                cached_payload["cached"] = True
                return cached_payload

        if progress_cb:
            progress_cb("Extracting verifiable factual claims from article...", 15)

        claims: List[ExtractedClaim] = extract_claims(news_text, max_claims=max_claims)
        claim_results: List[ClaimVerificationResult] = []
        all_sources_map: Dict[str, dict] = {}

        if not claims:
            if progress_cb:
                progress_cb("No checkworthy factual claims extracted.", 85)
        else:
            total_claims = len(claims)
            for idx, claim in enumerate(claims):
                pct_base = 20 + int((idx / max(1, total_claims)) * 65)
                if progress_cb:
                    progress_cb(
                        f"Searching external sources for claim {idx + 1}/{total_claims}...",
                        pct_base,
                    )

                articles = self._search_providers(
                    query=claim.search_query,
                    claim_text=claim.text,
                    max_results=4,
                )

                if progress_cb:
                    progress_cb(
                        f"Retrieving & verifying evidence for claim {idx + 1}/{total_claims}...",
                        min(90, pct_base + 10),
                    )

                retrieved: List[RetrievedEvidence] = self.retrieval_service.retrieve_evidence_for_claim(
                    claim_text=claim.text,
                    articles=articles,
                    max_sources=4,
                )

                claim_ver = self.verification_service.verify_claim(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    search_query=claim.search_query,
                    evidence_items=retrieved,
                )
                claim_results.append(claim_ver)

                # Collect unique source cards with their NLI stance
                for ev in (
                    claim_ver.supporting_evidence
                    + claim_ver.contradicting_evidence
                    + claim_ver.neutral_evidence
                ):
                    key = ev.url.strip().lower().rstrip("/")
                    if key not in all_sources_map:
                        all_sources_map[key] = ev.to_dict()

        if progress_cb:
            progress_cb("Computing independent final assessment...", 95)

        final_assessment: FinalAssessment = compute_final_assessment(
            ml_prediction=ml_result["prediction"],
            ml_confidence=ml_result["confidence"],
            claim_results=claim_results,
        )

        elapsed = round(time.time() - start_ts, 3)

        payload = {
            "success": True,
            "cached": False,
            "verification_time": elapsed,
            "nli_model": self.verification_service.model_name,
            "ml_analysis": ml_result,
            "final_assessment": final_assessment.to_dict(),
            "claims": [c.to_dict() for c in claim_results],
            "sources": list(all_sources_map.values()),
        }

        with self._lock:
            self._result_cache[text_hash] = (time.time(), dict(payload))

        if progress_cb:
            progress_cb("External verification complete.", 100)

        return payload

    def start_background_job(self, news_text: str, ml_result: dict) -> str:
        """Queue external verification in the background and return `job_id`."""
        job_id = str(uuid.uuid4())
        with self._lock:
            # Prune old completed jobs (> 30 mins)
            cutoff = time.time() - 1800
            stale = [jid for jid, j in self._jobs.items() if j.get("created_at", 0) < cutoff]
            for jid in stale:
                self._jobs.pop(jid, None)

            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "progress": 5,
                "stage": "Initializing external verification pipeline...",
                "created_at": time.time(),
                "ml_analysis": ml_result,
                "result": None,
                "error": None,
            }

        def _update_progress(stage: str, pct: int):
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["stage"] = stage
                    self._jobs[job_id]["progress"] = pct

        def _worker():
            try:
                result = self.verify_article(
                    news_text=news_text,
                    ml_result=ml_result,
                    progress_cb=_update_progress,
                )
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["status"] = "completed"
                        self._jobs[job_id]["progress"] = 100
                        self._jobs[job_id]["stage"] = "External verification complete."
                        self._jobs[job_id]["result"] = result
            except Exception as exc:
                logger.exception("Background verification job %s failed", job_id)
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["status"] = "failed"
                        self._jobs[job_id]["error"] = str(exc)

        self._executor.submit(_worker)
        return job_id

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Return current job state dict or None if unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None
