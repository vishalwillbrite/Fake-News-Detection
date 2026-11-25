"""
search_provider.py
Abstract provider interface and shared data structures for external news search.
Allows additional fact-checking or news search providers to be plugged in cleanly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class SearchArticle:
    title: str
    url: str
    domain: str
    publisher: str
    published_date: Optional[str] = None
    language: Optional[str] = None
    provider: str = "gdelt"
    relevance_score: float = 0.0

    def to_dict(self):
        return asdict(self)


class NewsSearchProvider(ABC):
    """Abstract base class for external news and fact-check search providers."""

    name: str = "base"

    @abstractmethod
    def search(self, query: str, claim_text: str = "", max_results: int = 5) -> List[SearchArticle]:
        """Search external sources for articles relevant to `query` / `claim_text`."""
        raise NotImplementedError
