"""
claim_extraction_service.py
Extracts individual verifiable factual claims and concise search queries
from submitted news articles.
"""

from dataclasses import dataclass, asdict
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Common abbreviations that should not trigger sentence splits
_ABBREVIATIONS = (
    "U.S.", "U.K.", "E.U.", "U.N.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
    "Sen.", "Rep.", "Gov.", "Pres.", "Gen.", "Col.", "Lt.", "Sgt.",
    "Inc.", "Ltd.", "Corp.", "Co.", "St.", "Ave.", "Blvd.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec.",
    "e.g.", "i.e.", "vs.", "etc.", "No.", "Vol.",
)

# Low-signal boilerplate / non-factual phrases
_BOILERPLATE_PATTERNS = re.compile(
    r"\b(click here|subscribe now|sign up|read more|follow us|share this|"
    r"watch live|breaking news alert|all rights reserved|privacy policy|"
    r"terms of service|advertisement)\b",
    re.IGNORECASE,
)

# Subjective / pure opinion starters
_OPINION_STARTERS = re.compile(
    r"^(i think|i believe|in my opinion|personally|maybe|perhaps|let's|please)\b",
    re.IGNORECASE,
)

_FACTUAL_VERBS = {
    "announced", "launched", "signed", "confirmed", "passed", "approved",
    "rejected", "reported", "discovered", "revealed", "published", "released",
    "issued", "banned", "ordered", "elected", "appointed", "charged",
    "arrested", "convicted", "acquired", "merged", "invested", "increased",
    "decreased", "fell", "rose", "reached", "achieved", "established",
    "built", "opened", "closed", "warned", "stated", "said", "found",
    "showed", "recorded", "won", "lost", "voted", "enacted", "deployed",
}

_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "that",
    "this", "these", "those", "it", "its", "they", "their", "them", "he",
    "she", "his", "her", "we", "our", "who", "whom", "which", "what",
    "when", "where", "why", "how", "also", "more", "most", "other", "some",
    "such", "into", "over", "after", "before", "under", "between", "about",
    "according", "said", "told", "reported", "officials", "sources",
}


@dataclass
class ExtractedClaim:
    claim_id: int
    text: str
    search_query: str
    score: float

    def to_dict(self):
        return asdict(self)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences while protecting common abbreviations."""
    if not text:
        return []

    protected = text
    placeholders = {}
    for idx, abbr in enumerate(_ABBREVIATIONS):
        token = f"__ABBR_{idx}__"
        if abbr in protected:
            protected = protected.replace(abbr, token)
            placeholders[token] = abbr

    # Normalize newlines into sentence breaks
    protected = re.sub(r"[\r\n]+", ". ", protected)
    raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])", protected)

    sentences = []
    for part in raw_parts:
        restored = part
        for token, abbr in placeholders.items():
            restored = restored.replace(token, abbr)
        restored = re.sub(r"\s+", " ", restored).strip(" \t\n\r\"'•-")
        if restored:
            sentences.append(restored)
    return sentences


def _token_set(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _QUERY_STOPWORDS and len(t) > 2}


def _jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_search_query(claim_text: str, max_terms: int = 6) -> str:
    """
    Build a concise keyword query from a factual claim for external news search.
    Prioritizes capitalized entities, numbers/years, and non-stopword content words.
    """
    # Extract capitalized multi-word or single-word entities (skip very first word if generic)
    words = re.findall(r"[A-Za-z0-9'-]+", claim_text)
    entities = []
    numbers = []
    content_words = []

    for idx, word in enumerate(words):
        clean = word.strip("'-")
        if len(clean) < 2:
            continue
        low = clean.lower()
        if low in _QUERY_STOPWORDS:
            continue

        if re.fullmatch(r"\d{4}|\d+(?:\.\d+)?%?", clean):
            if clean not in numbers:
                numbers.append(clean)
        elif clean[0].isupper() and idx > 0:
            if clean not in entities:
                entities.append(clean)
        elif len(clean) >= 4:
            if clean not in content_words:
                content_words.append(clean)

    # Also include first word if it looks like a proper noun (all caps or capitalized acronym)
    if words:
        first = words[0].strip("'-")
        if len(first) >= 2 and first.lower() not in _QUERY_STOPWORDS and (first.isupper() or first[0].isupper()):
            if first not in entities and first not in content_words:
                entities.insert(0, first)

    selected = []
    for bucket in (entities, numbers, content_words):
        for term in bucket:
            if term.lower() not in {s.lower() for s in selected}:
                selected.append(term)
            if len(selected) >= max_terms:
                break
        if len(selected) >= max_terms:
            break

    return " ".join(selected)


def _score_claim(sentence: str) -> float:
    """Heuristic checkworthiness score for a candidate sentence."""
    score = 0.2
    words = sentence.split()

    # Prefer well-formed medium-length factual statements
    if 8 <= len(words) <= 35:
        score += 0.25
    elif 6 <= len(words) <= 50:
        score += 0.1

    # Numbers, percentages, years
    if re.search(r"\b\d+(?:\.\d+)?%?\b", sentence):
        score += 0.2

    # Proper nouns beyond first word
    proper_nouns = re.findall(r"(?<!^)\b[A-Z][a-zA-Z]{2,}\b", sentence)
    if proper_nouns:
        score += min(0.25, 0.08 * len(proper_nouns))

    # Factual reporting/action verbs
    lower_words = {w.strip(".,;:!?\"'").lower() for w in words}
    if lower_words & _FACTUAL_VERBS:
        score += 0.2

    return round(min(score, 1.0), 3)


def extract_claims(article_text: str, max_claims: int = 4) -> List[ExtractedClaim]:
    """
    Extract up to `max_claims` verifiable factual claims from an article.
    Preserves order of appearance after selecting top checkworthy, non-duplicate claims.
    """
    if not article_text or not article_text.strip():
        return []

    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", article_text).strip()
    sentences = _split_sentences(cleaned)

    candidates = []
    for idx, sent in enumerate(sentences):
        words = sent.split()
        if len(words) < 5 or len(sent) < 25:
            continue
        if sent.endswith("?"):
            continue
        if _BOILERPLATE_PATTERNS.search(sent):
            continue
        if _OPINION_STARTERS.match(sent):
            continue

        query = build_search_query(sent)
        if len(query.split()) < 2:
            continue

        score = _score_claim(sent)
        candidates.append((idx, sent, query, score, _token_set(sent)))

    # Fallback: if no sentence passed strict filters (e.g., short headline), use the cleaned text directly
    if not candidates and len(cleaned.split()) >= 3:
        short_text = " ".join(cleaned.split()[:40])
        query = build_search_query(short_text) or " ".join(
            [w for w in re.findall(r"[A-Za-z0-9]+", short_text) if w.lower() not in _QUERY_STOPWORDS][:6]
        )
        if query:
            return [ExtractedClaim(claim_id=1, text=short_text, search_query=query, score=0.5)]
        return []

    # Sort by score descending to pick most checkworthy, deduplicating similar claims
    ranked = sorted(candidates, key=lambda x: x[3], reverse=True)
    selected = []
    for item in ranked:
        if any(_jaccard_similarity(item[4], s[4]) > 0.65 for s in selected):
            continue
        selected.append(item)
        if len(selected) >= max_claims:
            break

    # Restore original article order
    selected.sort(key=lambda x: x[0])

    claims = [
        ExtractedClaim(
            claim_id=i + 1,
            text=item[1],
            search_query=item[2],
            score=item[3],
        )
        for i, item in enumerate(selected)
    ]
    logger.info("Extracted %d factual claims from article (%d chars)", len(claims), len(article_text))
    return claims
