# sec_rag/similarity/explainer.py

from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI

class SimilarityExplainer:
    """Generate natural language explanations of company similarities."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3
        )
    
    def explain_similarity(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str,
        chunks1: List[str],
        chunks2: List[str]
    ) -> str:
        """
        Generate explanation of why two companies are similar.
        """
        prompt = f"""
        You are analyzing similarity between two companies based on their 10-K filings.
        
        Compare {ticker1} and {ticker2} in terms of: {dimension}
        
        {ticker1} Information:
        {self._format_chunks(chunks1)}
        
        {ticker2} Information:
        {self._format_chunks(chunks2)}
        
        Provide a concise 2-3 sentence explanation of:
        1. What specific similarities exist
        2. Why this makes them comparable
        3. Any notable differences
        
        Be specific and cite concrete details from the filings.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _format_chunks(self, chunks: List[str]) -> str:
        """Format chunks for prompt."""
        formatted = []
        for i, chunk in enumerate(chunks[:3], 1):
            formatted.append(f"Excerpt {i}: {chunk[:300]}...")
        return "\n\n".join(formatted)