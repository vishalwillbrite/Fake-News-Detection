"""
final_decision_service.py
Combines the independent ML writing-style prediction with the independent
external evidence verification assessment into a transparent final verdict.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

from services.evidence_verification_service import ClaimVerificationResult


@dataclass
class FinalAssessment:
    final_label: str  # REAL | FAKE | MIXED / NEEDS REVIEW | UNVERIFIED
    final_class: str  # real | fake | mixed | unverified
    external_verdict: str  # SUPPORTED | CONTRADICTED | MIXED | INSUFFICIENT EVIDENCE
    ml_label: str
    ml_confidence: float
    ml_external_disagreement: bool
    disagreement_note: Optional[str]
    summary: str
    claim_counts: dict

    def to_dict(self):
        return asdict(self)


def compute_final_assessment(
    ml_prediction: int,
    ml_confidence: float,
    claim_results: List[ClaimVerificationResult],
) -> FinalAssessment:
    """
    Produce final article assessment from independent ML prediction and
    independent external NLI claim verifications.
    Never lets ML probability alone determine factual truth.
    """
    ml_label = "Real News" if int(ml_prediction) == 1 else "Fake News"

    counts = {
        "total": len(claim_results),
        "supported": sum(1 for c in claim_results if c.label == "SUPPORTED"),
        "contradicted": sum(1 for c in claim_results if c.label == "CONTRADICTED"),
        "mixed": sum(1 for c in claim_results if c.label == "MIXED"),
        "insufficient": sum(1 for c in claim_results if c.label == "INSUFFICIENT EVIDENCE"),
    }

    # Determine independent external evidence verdict
    if counts["total"] == 0 or (
        counts["supported"] == 0 and counts["contradicted"] == 0 and counts["mixed"] == 0
    ):
        external_verdict = "INSUFFICIENT EVIDENCE"
    elif counts["mixed"] > 0 or (counts["supported"] > 0 and counts["contradicted"] > 0):
        external_verdict = "MIXED"
    elif counts["contradicted"] > 0:
        external_verdict = "CONTRADICTED"
    else:
        external_verdict = "SUPPORTED"

    disagreement = False
    disagreement_note: Optional[str] = None

    if external_verdict == "INSUFFICIENT EVIDENCE":
        final_label = "UNVERIFIED"
        final_class = "unverified"
        summary = (
            "Insufficient reliable external evidence was retrieved to confirm or refute the "
            "extracted claims. Missing evidence does not imply fabrication; the independent "
            f"ML linguistic model classified the text style as {ml_label} ({ml_confidence}%)."
        )
    elif external_verdict == "MIXED":
        final_label = "MIXED / NEEDS REVIEW"
        final_class = "mixed"
        summary = (
            "External sources present conflicting or mixed evidence across the extracted claims. "
            "Careful editorial review of the supporting and contradicting sources is recommended."
        )
    elif external_verdict == "SUPPORTED":
        if int(ml_prediction) == 1:
            final_label = "REAL"
            final_class = "real"
            summary = (
                f"Central factual claims ({counts['supported']} of {counts['total']}) are supported "
                "by retrieved external news sources, aligning with the ML linguistic assessment."
            )
        else:
            disagreement = True
            final_label = "MIXED / NEEDS REVIEW"
            final_class = "mixed"
            disagreement_note = (
                f"Model Disagreement: External sources support {counts['supported']} extracted claim(s), "
                f"whereas the ML writing-style classifier predicted Fake News ({ml_confidence}%). "
                "Human review is recommended."
            )
            summary = disagreement_note
    else:  # external_verdict == "CONTRADICTED"
        if int(ml_prediction) == 0:
            final_label = "FAKE"
            final_class = "fake"
            summary = (
                f"Material claims ({counts['contradicted']} of {counts['total']}) are contradicted "
                "by retrieved external news sources, aligning with the ML linguistic assessment."
            )
        else:
            disagreement = True
            final_label = "MIXED / NEEDS REVIEW"
            final_class = "mixed"
            disagreement_note = (
                f"Model Disagreement: External sources contradict {counts['contradicted']} claim(s), "
                f"whereas the ML writing-style classifier predicted Real News ({ml_confidence}%). "
                "Human review is recommended."
            )
            summary = disagreement_note

    return FinalAssessment(
        final_label=final_label,
        final_class=final_class,
        external_verdict=external_verdict,
        ml_label=ml_label,
        ml_confidence=ml_confidence,
        ml_external_disagreement=disagreement,
        disagreement_note=disagreement_note,
        summary=summary,
        claim_counts=counts,
    )
