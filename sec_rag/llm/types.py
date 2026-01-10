"""
Data types for LLM explanations.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class SimilarityExplanation:
    """Structured explanation of company similarity."""
    ticker1: str
    ticker2: str
    dimension: str
    explanation: str
    key_similarities: List[str]
    key_differences: List[str]
    confidence: str