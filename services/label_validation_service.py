"""
label_validation_service.py
Validates candidate news articles for reliable ground-truth labels.
Never assumes GDELT articles are REAL, never uses ML predictions as labels,
and never infers an entire article's label from an unrelated single claim.
Places articles without trustworthy labels into NEEDS_REVIEW.
"""

from dataclasses import dataclass
import json
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_FALSE_RATING_KEYWORDS = {
    "false", "pants on fire", "fabricated", "fake", "hoax", "altered",
    "misleading", "mostly false", "incorrect", "satire", "unproven",
}
_TRUE_RATING_KEYWORDS = {
    "true", "mostly true", "accurate", "correct", "verified",
}


@dataclass
class LabelValidationResult:
    proposed_label: Optional[int]  # 0=Fake, 1=Real, None=Unknown
    status: str  # PENDING | NEEDS_REVIEW
    label_source: str  # claim_review_fact_check | unverified
    evidence: dict
    source_url: Optional[str] = None


def _extract_claim_review_from_jsonld(html: str) -> Optional[dict]:
    """
    Inspect JSON-LD blocks in HTML for schema.org ClaimReview fact-check verdicts.
    Only trusts explicit structured ClaimReview ratings.
    """
    if not html:
        return None
    try:
        soup = BeautifulSoup(html[:120000], "html.parser")
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (script.string or "").strip()
            if not raw or "ClaimReview" not in raw:
                continue
            try:
                parsed = json.loads(raw)
            except ValueError:
                continue

            nodes = parsed if isinstance(parsed, list) else [parsed]
            # Handle @graph container
            expanded = []
            for n in nodes:
                if isinstance(n, dict) and isinstance(n.get("@graph"), list):
                    expanded.extend(n["@graph"])
                elif isinstance(n, dict):
                    expanded.append(n)

            for node in expanded:
                if not isinstance(node, dict):
                    continue
                node_type = node.get("@type", "")
                types = node_type if isinstance(node_type, list) else [node_type]
                if "ClaimReview" not in types:
                    continue

                rating_obj = node.get("reviewRating") or {}
                if not isinstance(rating_obj, dict):
                    continue
                alt_name = str(rating_obj.get("alternateName") or "").strip()
                if not alt_name:
                    continue

                return {
                    "claim_reviewed": str(node.get("claimReviewed") or "").strip(),
                    "rating": alt_name,
                    "author": (
                        node.get("author", {}).get("name")
                        if isinstance(node.get("author"), dict)
                        else str(node.get("author") or "")
                    ),
                    "url": str(node.get("url") or "").strip(),
                }
    except Exception as exc:
        logger.debug("JSON-LD ClaimReview parse failed: %s", exc)
    return None


def validate_candidate_label(
    title: str,
    content: str,
    url: str,
    raw_html: str = "",
) -> LabelValidationResult:
    """
    Attempt to establish a trustworthy label from explicit structured fact-check
    metadata (schema.org ClaimReview).
    - If a clear ClaimReview verdict is found -> PENDING with proposed_label (0 or 1).
    - Otherwise -> NEEDS_REVIEW with proposed_label=None (human-review queue).
    """
    claim_review = _extract_claim_review_from_jsonld(raw_html)
    if claim_review:
        rating_norm = re.sub(r"\s+", " ", claim_review["rating"].lower()).strip()
        if any(k in rating_norm for k in _FALSE_RATING_KEYWORDS):
            return LabelValidationResult(
                proposed_label=0,
                status="PENDING",
                label_source="claim_review_fact_check",
                evidence={
                    "method": "ClaimReview JSON-LD",
                    "rating": claim_review["rating"],
                    "claim_reviewed": claim_review["claim_reviewed"],
                    "reviewer": claim_review["author"],
                },
                source_url=claim_review.get("url") or url,
            )
        if rating_norm in _TRUE_RATING_KEYWORDS:
            return LabelValidationResult(
                proposed_label=1,
                status="PENDING",
                label_source="claim_review_fact_check",
                evidence={
                    "method": "ClaimReview JSON-LD",
                    "rating": claim_review["rating"],
                    "claim_reviewed": claim_review["claim_reviewed"],
                    "reviewer": claim_review["author"],
                },
                source_url=claim_review.get("url") or url,
            )

    # No trustworthy ground-truth label could be established automatically
    return LabelValidationResult(
        proposed_label=None,
        status="NEEDS_REVIEW",
        label_source="unverified",
        evidence={
            "method": "No structured ClaimReview verdict found",
            "note": "Queued for human editorial review before entering training data.",
        },
        source_url=url,
    )
