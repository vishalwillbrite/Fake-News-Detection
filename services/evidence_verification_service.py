"""
evidence_verification_service.py
Local lightweight Natural Language Inference (NLI) verification service.
Uses `cross-encoder/nli-deberta-v3-xsmall` (22M params, CPU-friendly) to compare
extracted claims against retrieved external evidence passages.
"""

from dataclasses import dataclass, asdict
import logging
import threading
from typing import Callable, Dict, List, Optional

from services.evidence_retrieval_service import RetrievedEvidence

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"

ENTAILMENT_THRESHOLD = 0.58
CONTRADICTION_THRESHOLD = 0.58
MARGIN_THRESHOLD = 0.15


@dataclass
class EvaluatedEvidence:
    title: str
    url: str
    domain: str
    publisher: str
    published_date: Optional[str]
    evidence_passage: str
    accessible: bool
    status_note: str
    stance: str  # SUPPORTING | CONTRADICTING | NEUTRAL
    entailment_score: float
    contradiction_score: float
    neutral_score: float

    def to_dict(self):
        return asdict(self)


@dataclass
class ClaimVerificationResult:
    claim_id: int
    claim_text: str
    search_query: str
    label: str  # SUPPORTED | CONTRADICTED | MIXED | INSUFFICIENT EVIDENCE
    rationale: str
    supporting_evidence: List[EvaluatedEvidence]
    contradicting_evidence: List[EvaluatedEvidence]
    neutral_evidence: List[EvaluatedEvidence]
    unretrieved_sources: List[EvaluatedEvidence]
    sources_checked: int
    search_note: str = ""

    def to_dict(self):
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "search_query": self.search_query,
            "label": self.label,
            "rationale": self.rationale,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "contradicting_evidence": [e.to_dict() for e in self.contradicting_evidence],
            "neutral_evidence": [e.to_dict() for e in self.neutral_evidence],
            "unretrieved_sources": [e.to_dict() for e in self.unretrieved_sources],
            "sources_checked": self.sources_checked,
            "search_note": self.search_note,
        }


class EvidenceVerificationService:
    """
    Evaluates factual claims against external evidence passages using a local NLI model.
    Supports dependency-injected `nli_scorer` for deterministic unit testing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        nli_scorer: Optional[Callable[[str, str], Dict[str, float]]] = None,
    ):
        self.model_name = model_name
        self._custom_scorer = nli_scorer
        self._model = None
        self._tokenizer = None
        self._label_map: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    def _ensure_model_loaded(self):
        if self._custom_scorer is not None:
            return
        if self._model is not None and self._tokenizer is not None:
            return

        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                logger.info("Loading local NLI model '%s'...", self.model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.eval()

                # Inspect id2label dynamically so label indices are always accurate
                raw_id2label = getattr(self._model.config, "id2label", {}) or {}
                self._label_map = {}
                for idx, lbl in raw_id2label.items():
                    norm = str(lbl).strip().lower()
                    if "entail" in norm:
                        self._label_map[int(idx)] = "entailment"
                    elif "contra" in norm:
                        self._label_map[int(idx)] = "contradiction"
                    else:
                        self._label_map[int(idx)] = "neutral"

                # Fallback standard DeBERTa-v3 NLI mapping if config lacks labels
                if not self._label_map:
                    self._label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}

                logger.info("Local NLI model loaded successfully (labels: %s)", self._label_map)
            except Exception as exc:
                self._load_error = str(exc)
                logger.error("Failed to load local NLI model '%s': %s", self.model_name, exc)
                raise

    def score_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute NLI alignment probabilities for (premise=evidence, hypothesis=claim).
        Returns {'entailment': float, 'contradiction': float, 'neutral': float}.
        Note: These are NLI semantic alignment scores, NOT factual truth probabilities.
        """
        if not premise or not hypothesis:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

        if self._custom_scorer is not None:
            return self._custom_scorer(premise, hypothesis)

        self._ensure_model_loaded()

        import torch

        inputs = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for idx, p in enumerate(probs):
            key = self._label_map.get(idx, "neutral")
            scores[key] += float(p)
        return {k: round(v, 4) for k, v in scores.items()}

    def verify_claim(
        self,
        claim_id: int,
        claim_text: str,
        search_query: str,
        evidence_items: List[RetrievedEvidence],
        search_note: str = "",
    ) -> ClaimVerificationResult:
        """
        Compare a single extracted claim against retrieved external evidence items.
        Only evaluates articles whose content was actually retrieved; inaccessible
        articles are recorded in `unretrieved_sources` and never used as proof.
        """
        if not evidence_items:
            return ClaimVerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                search_query=search_query,
                label="INSUFFICIENT EVIDENCE",
                rationale=(
                    "No relevant external news articles were returned by search for this claim. "
                    "Missing evidence does not prove the claim is false."
                ),
                supporting_evidence=[],
                contradicting_evidence=[],
                neutral_evidence=[],
                unretrieved_sources=[],
                sources_checked=0,
                search_note=search_note,
            )

        supporting: List[EvaluatedEvidence] = []
        contradicting: List[EvaluatedEvidence] = []
        neutral: List[EvaluatedEvidence] = []
        unretrieved: List[EvaluatedEvidence] = []

        for ev in evidence_items:
            premise = (ev.evidence_passage or "").strip()
            if not ev.accessible or not premise:
                unretrieved.append(
                    EvaluatedEvidence(
                        title=ev.title,
                        url=ev.url,
                        domain=ev.domain,
                        publisher=ev.publisher,
                        published_date=ev.published_date,
                        evidence_passage="",
                        accessible=False,
                        status_note=ev.status_note or "Content could not be retrieved",
                        stance="UNRETRIEVED",
                        entailment_score=0.0,
                        contradiction_score=0.0,
                        neutral_score=0.0,
                    )
                )
                continue

            try:
                scores = self.score_nli(premise, claim_text)
            except Exception as exc:
                logger.warning("NLI scoring error for '%s': %s", ev.url, exc)
                scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

            ent = scores.get("entailment", 0.0)
            con = scores.get("contradiction", 0.0)
            neu = scores.get("neutral", 0.0)

            if ent >= ENTAILMENT_THRESHOLD and (ent - con) >= MARGIN_THRESHOLD:
                stance = "SUPPORTING"
            elif con >= CONTRADICTION_THRESHOLD and (con - ent) >= MARGIN_THRESHOLD:
                stance = "CONTRADICTING"
            else:
                stance = "NEUTRAL"

            evaluated = EvaluatedEvidence(
                title=ev.title,
                url=ev.url,
                domain=ev.domain,
                publisher=ev.publisher,
                published_date=ev.published_date,
                evidence_passage=premise,
                accessible=True,
                status_note=ev.status_note,
                stance=stance,
                entailment_score=round(ent, 3),
                contradiction_score=round(con, 3),
                neutral_score=round(neu, 3),
            )

            if stance == "SUPPORTING":
                supporting.append(evaluated)
            elif stance == "CONTRADICTING":
                contradicting.append(evaluated)
            else:
                neutral.append(evaluated)

        accessible_count = len(supporting) + len(contradicting) + len(neutral)
        unretrieved_note = (
            f" ({len(unretrieved)} additional candidate source(s) could not be accessed.)"
            if unretrieved
            else ""
        )

        # Determine claim-level label strictly from accessible retrieved content
        if supporting and contradicting:
            label = "MIXED"
            rationale = (
                f"Conflicting evidence across {accessible_count} retrieved article(s): "
                f"{len(supporting)} source(s) entail the claim while {len(contradicting)} "
                f"source(s) contradict it.{unretrieved_note}"
            )
        elif supporting:
            label = "SUPPORTED"
            top_pubs = ", ".join(e.publisher for e in supporting[:3])
            rationale = (
                f"Supported by {len(supporting)} retrieved article(s) ({top_pubs}), whose "
                f"extracted passages semantically entail this claim.{unretrieved_note}"
            )
        elif contradicting:
            label = "CONTRADICTED"
            top_pubs = ", ".join(e.publisher for e in contradicting[:3])
            rationale = (
                f"Contradicted by {len(contradicting)} retrieved article(s) ({top_pubs}), whose "
                f"extracted passages conflict with this claim.{unretrieved_note}"
            )
        elif accessible_count > 0:
            label = "INSUFFICIENT EVIDENCE"
            rationale = (
                f"Examined {accessible_count} accessible article(s), but the extracted passages "
                f"were neutral or inconclusive and did not clearly confirm or refute the claim.{unretrieved_note}"
            )
        else:
            label = "INSUFFICIENT EVIDENCE"
            rationale = (
                f"Found {len(unretrieved)} candidate article(s), but none were accessible for "
                "content extraction, so the claim cannot be verified from retrieved text."
            )

        return ClaimVerificationResult(
            claim_id=claim_id,
            claim_text=claim_text,
            search_query=search_query,
            label=label,
            rationale=rationale,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            neutral_evidence=neutral,
            unretrieved_sources=unretrieved,
            sources_checked=len(evidence_items),
            search_note=search_note,
        )
