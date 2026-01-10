"""
LLM-powered explanations for company similarity analysis.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class SimilarityExplainer:
    """
    Generate natural language explanations for company similarities.
    Uses LLM to analyze and explain why companies are similar or different.
    """
    
    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3
    ):
        """
        Initialize the explainer.
        
        Args:
            vector_store: ChromaDB store for retrieving company data
            model: LLM model to use for explanations
            temperature: LLM temperature (lower = more focused)
        """
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature
        )
        logger.info(f"Initialized SimilarityExplainer with {model}")
    
    def explain_similarity(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str,
        max_chunks: int = 5
    ) -> SimilarityExplanation:
        """
        Generate explanation for why two companies are similar in a dimension.
        
        Args:
            ticker1: First company ticker
            ticker2: Second company ticker
            dimension: Dimension to explain (business_model, risk_profile, etc.)
            max_chunks: Maximum chunks to use per company
        
        Returns:
            SimilarityExplanation with natural language explanation
        """
        logger.info(f"Generating explanation: {ticker1} vs {ticker2} ({dimension})")
        
        # Get relevant chunks for both companies
        chunks1 = self._get_relevant_chunks(ticker1, dimension, max_chunks)
        chunks2 = self._get_relevant_chunks(ticker2, dimension, max_chunks)
        
        if not chunks1 or not chunks2:
            logger.warning(f"Insufficient data for {ticker1} or {ticker2}")
            return self._create_insufficient_data_explanation(
                ticker1, ticker2, dimension
            )
        
        # Generate explanation using LLM
        explanation_text = self._generate_llm_explanation(
            ticker1, ticker2, dimension, chunks1, chunks2
        )
        
        # Parse the structured response
        parsed = self._parse_explanation(explanation_text)
        
        return SimilarityExplanation(
            ticker1=ticker1,
            ticker2=ticker2,
            dimension=dimension,
            explanation=parsed['explanation'],
            key_similarities=parsed['similarities'],
            key_differences=parsed['differences'],
            confidence=parsed['confidence']
        )
    
    def explain_overall_similarity(
        self,
        ticker1: str,
        ticker2: str,
        dimension_scores: Dict[str, float]
    ) -> str:
        """
        Generate overall explanation across all dimensions.
        
        Args:
            ticker1: First company ticker
            ticker2: Second company ticker  
            dimension_scores: Scores for each dimension
        
        Returns:
            Natural language explanation of overall similarity
        """
        logger.info(f"Generating overall explanation: {ticker1} vs {ticker2}")
        
        # Get the top dimensions by score
        top_dimensions = sorted(
            dimension_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Build explanation for top dimensions
        dimension_details = []
        for dim, score in top_dimensions:
            chunks1 = self._get_relevant_chunks(ticker1, dim, 3)
            chunks2 = self._get_relevant_chunks(ticker2, dim, 3)
            
            if chunks1 and chunks2:
                detail = f"{dim} (score: {score:.2f})"
                dimension_details.append(detail)
        
        prompt = f"""
        Compare {ticker1} and {ticker2} based on their 10-K filings.
        
        Overall similarity score breakdown:
        {chr(10).join(f"- {d}" for d in dimension_details)}
        
        Provide a 3-4 sentence executive summary explaining:
        1. What makes these companies fundamentally similar
        2. The key dimensions where they align
        3. Any important differences to note
        
        Be specific and cite concrete business characteristics.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _get_relevant_chunks(
        self,
        ticker: str,
        dimension: str,
        max_chunks: int
    ) -> List[str]:
        """
        Retrieve relevant chunks for a company and dimension.
        """
        if not self.vector_store:
            logger.warning("No vector store provided")
            return []
        
        # Dimension-specific queries
        dimension_queries = {
            'business_model': 'revenue sources business segments products services',
            'risk_profile': 'risks challenges regulatory compliance competition',
            'financial_structure': 'debt capital liquidity cash flow profitability',
            'geographic_footprint': 'international markets regions countries global',
            'legal_matters': 'litigation legal proceedings lawsuits settlements'
        }
        
        query = dimension_queries.get(dimension, dimension)
        
        try:
            results = self.vector_store.similarity_search(
                query,
                k=max_chunks,
                filter={"ticker": ticker}
            )
            
            chunks = [doc.page_content for doc in results]
            logger.debug(f"Retrieved {len(chunks)} chunks for {ticker}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _generate_llm_explanation(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str,
        chunks1: List[str],
        chunks2: List[str]
    ) -> str:
        """
        Use LLM to generate structured explanation.
        """
        # Format chunks for prompt
        ticker1_text = self._format_chunks(chunks1, max_chars=1500)
        ticker2_text = self._format_chunks(chunks2, max_chars=1500)
        
        # Dimension-specific context
        dimension_context = {
            'business_model': 'how they generate revenue, their products/services, and customer base',
            'risk_profile': 'the risks they face (regulatory, competitive, operational, etc.)',
            'financial_structure': 'their capital structure, debt levels, profitability, and cash flow',
            'geographic_footprint': 'their international presence and geographic markets',
            'legal_matters': 'their legal proceedings, litigation, and regulatory compliance'
        }
        
        context = dimension_context.get(dimension, dimension.replace('_', ' '))
        
        prompt = f"""
        You are analyzing why two companies are similar based on their SEC 10-K filings.
        
        Compare {ticker1} and {ticker2} in terms of: {context}
        
        {ticker1} Information from 10-K:
        {ticker1_text}
        
        {ticker2} Information from 10-K:
        {ticker2_text}
        
        Provide your analysis in this EXACT format:
        
        EXPLANATION:
        [2-3 sentences explaining what makes them similar in this dimension. Be specific and cite concrete details from the filings.]
        
        KEY SIMILARITIES:
        - [Similarity 1: Be specific]
        - [Similarity 2: Be specific]
        - [Similarity 3: Be specific]
        
        KEY DIFFERENCES:
        - [Difference 1: Be specific]
        - [Difference 2: Be specific]
        
        CONFIDENCE:
        [high/medium/low] - [One sentence explaining why this confidence level]
        
        Be concise, specific, and cite concrete details from the filings.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _format_chunks(self, chunks: List[str], max_chars: int = 1500) -> str:
        """
        Format chunks for inclusion in prompt.
        """
        formatted = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks, 1):
            if total_chars >= max_chars:
                break
            
            chunk_text = chunk[:500]  # Limit each chunk
            formatted.append(f"Excerpt {i}:\n{chunk_text}\n")
            total_chars += len(chunk_text)
        
        return "\n".join(formatted)
    
    def _parse_explanation(self, text: str) -> Dict:
        """
        Parse the structured LLM response.
        """
        lines = text.strip().split('\n')
        
        result = {
            'explanation': '',
            'similarities': [],
            'differences': [],
            'confidence': 'medium'
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                continue
            elif line.startswith('KEY SIMILARITIES:'):
                current_section = 'similarities'
                continue
            elif line.startswith('KEY DIFFERENCES:'):
                current_section = 'differences'
                continue
            elif line.startswith('CONFIDENCE:'):
                current_section = 'confidence'
                # Extract confidence level
                if 'high' in line.lower():
                    result['confidence'] = 'high'
                elif 'low' in line.lower():
                    result['confidence'] = 'low'
                else:
                    result['confidence'] = 'medium'
                continue
            
            # Add content to current section
            if current_section == 'explanation':
                result['explanation'] += line + ' '
            elif current_section == 'similarities' and line.startswith('-'):
                result['similarities'].append(line[1:].strip())
            elif current_section == 'differences' and line.startswith('-'):
                result['differences'].append(line[1:].strip())
        
        # Clean up explanation
        result['explanation'] = result['explanation'].strip()
        
        return result
    
    def _create_insufficient_data_explanation(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str
    ) -> SimilarityExplanation:
        """
        Create explanation when data is insufficient.
        """
        return SimilarityExplanation(
            ticker1=ticker1,
            ticker2=ticker2,
            dimension=dimension,
            explanation=f"Insufficient data available to compare {ticker1} and {ticker2} in the {dimension} dimension.",
            key_similarities=["Unable to determine - insufficient data"],
            key_differences=["Unable to determine - insufficient data"],
            confidence="low"
        )

def create_explainer(vector_store: Chroma) -> SimilarityExplainer:
    """
    Factory function to create a SimilarityExplainer.
    
    Args:
        vector_store: ChromaDB vector store
    
    Returns:
        Configured SimilarityExplainer instance
    """
    return SimilarityExplainer(vector_store=vector_store)