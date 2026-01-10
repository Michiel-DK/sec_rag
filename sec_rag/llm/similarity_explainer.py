"""
LLM-powered explanations for company similarity analysis.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma


from .types import SimilarityExplanation  # CHANGED: Import from types
from .explanation_cache import ExplanationCache  # This now works!

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityExplainer:
    def __init__(
        self,
        vector_store: Optional[Chroma] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        requests_per_minute: int = 8,
        use_cache: bool = True,  # NEW
        cache_dir: str = "./outputs/explanations"  # NEW
    ):
        """
        Initialize the explainer.
        
        Args:
            vector_store: ChromaDB store for retrieving company data
            model: LLM model
            temperature: LLM temperature
            requests_per_minute: Rate limit
            use_cache: Whether to use caching (default: True)
            cache_dir: Directory for cached explanations
        """
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature
        )
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0
        
        # NEW: Initialize cache
        self.use_cache = use_cache
        self.cache = ExplanationCache(cache_dir) if use_cache else None
        
        logger.info(f"Initialized SimilarityExplainer with {model}")
        logger.info(f"Rate limit: {requests_per_minute} requests/minute ({self.min_delay:.1f}s between calls)")
        if use_cache:
            logger.info(f"Caching enabled: {cache_dir}")
    
    def explain_similarity(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str,
        max_chunks: int = 3,
        force_refresh: bool = False  # NEW: Force regeneration
    ) -> SimilarityExplanation:
        """
        Generate explanation for why two companies are similar in a dimension.
        Uses cache to avoid regenerating existing explanations.
        
        Args:
            ticker1: First company ticker
            ticker2: Second company ticker
            dimension: Dimension to explain
            max_chunks: Maximum chunks to use per company
            force_refresh: If True, regenerate even if cached
        
        Returns:
            SimilarityExplanation with natural language explanation
        """
        # Check cache first (unless force_refresh)
        if self.use_cache and not force_refresh:
            cached = self.cache.get_cached(ticker1, ticker2, dimension)
            if cached:
                logger.info(f"✓ Using cached explanation: {ticker1} vs {ticker2} ({dimension})")
                return cached
        
        logger.info(f"🔍 Generating explanation: {ticker1} vs {ticker2} ({dimension})")
        
        # Get relevant chunks for both companies
        chunks1 = self._get_relevant_chunks(ticker1, dimension, max_chunks)
        chunks2 = self._get_relevant_chunks(ticker2, dimension, max_chunks)
        
        if not chunks1 or not chunks2:
            logger.warning(f"⚠️  Insufficient data for {ticker1} or {ticker2}")
            explanation = self._create_insufficient_data_explanation(
                ticker1, ticker2, dimension
            )
        else:
            # Generate explanation using LLM (with rate limiting)
            try:
                explanation_text = self._generate_llm_explanation(
                    ticker1, ticker2, dimension, chunks1, chunks2
                )
                
                # Parse the structured response
                parsed = self._parse_explanation(explanation_text)
                
                explanation = SimilarityExplanation(
                    ticker1=ticker1,
                    ticker2=ticker2,
                    dimension=dimension,
                    explanation=parsed['explanation'],
                    key_similarities=parsed['similarities'],
                    key_differences=parsed['differences'],
                    confidence=parsed['confidence']
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to generate explanation: {e}")
                explanation = self._create_error_explanation(
                    ticker1, ticker2, dimension, str(e)
                )
        
        # Save to cache
        if self.use_cache:
            try:
                self.cache.save(explanation)
            except Exception as e:
                logger.warning(f"Failed to cache explanation: {e}")
        
        return explanation
    
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
        """Retrieve relevant chunks, focusing on substance."""
        
        if not self.vector_store:
            return []
        
        # Better queries that focus on substance
        dimension_queries = {
            'business_model': [
                f'{ticker} primary products sell customers',
                f'{ticker} how company generates revenue business',
                f'{ticker} competitive advantages market position',
                f'{ticker} customer base target market'
            ],
            'risk_profile': [
                f'{ticker} major risks challenges faces',
                f'{ticker} competitive threats market risks',
                f'{ticker} regulatory compliance risks',
                f'{ticker} operational technology risks'
            ],
            'financial_structure': [
                f'{ticker} debt obligations borrowing',
                f'{ticker} cash flow profitability',
                f'{ticker} capital structure financing',
                f'{ticker} dividend share repurchase'
            ],
            'geographic_footprint': [
                f'{ticker} international operations markets',
                f'{ticker} revenue by region geography',
                f'{ticker} foreign markets expansion',
                f'{ticker} global presence countries'
            ],
            'legal_matters': [
                f'{ticker} litigation lawsuits legal',
                f'{ticker} regulatory investigations',
                f'{ticker} legal proceedings disputes',
                f'{ticker} settlements fines penalties'
            ]
        }
        
        queries = dimension_queries.get(dimension, [dimension])
        
        # Get chunks from multiple queries
        all_chunks = []
        for query in queries[:2]:  # Use first 2 queries
            try:
                results = self.vector_store.similarity_search(
                    query,
                    k=max_chunks,
                    filter={"ticker": ticker}
                )
                all_chunks.extend([doc.page_content for doc in results])
            except:
                pass
        
        # Deduplicate and return
        unique_chunks = list(dict.fromkeys(all_chunks))
        return unique_chunks[:max_chunks]
    
    def _generate_llm_explanation(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str,
        chunks1: List[str],
        chunks2: List[str]
    ) -> str:
        """
        Use LLM to generate structured explanation focused on business substance.
        """
        # Format chunks for prompt
        ticker1_text = self._format_chunks(chunks1, max_chars=1500)
        ticker2_text = self._format_chunks(chunks2, max_chars=1500)
        
        # Dimension-specific context with business focus
        dimension_context = {
            'business_model': 'their actual business operations: what they sell, who buys it, how they make money, and what makes them unique',
            'risk_profile': 'the real business risks they face: competitive threats, regulatory challenges, operational vulnerabilities, and market dependencies',
            'financial_structure': 'their financial strategy: how they fund operations, manage debt, generate cash, and return value to shareholders',
            'geographic_footprint': 'their global presence: which markets they operate in, international revenue mix, and regional strategies',
            'legal_matters': 'their legal exposure: active litigation, regulatory investigations, compliance issues, and potential liabilities'
        }
        
        context = dimension_context.get(dimension, dimension.replace('_', ' '))
        
        prompt = f"""You are a business analyst comparing two companies based on their 10-K filings.

            Compare {ticker1} and {ticker2} focusing on: {context}

            CRITICAL INSTRUCTIONS:
            - Focus on BUSINESS SUBSTANCE, not 10-K structure or accounting policies
            - Ignore generic statements like "revenue is recognized when control transfers"
            - Look for what makes these companies' ACTUAL BUSINESSES similar or different
            - If the excerpts lack substance, use your knowledge of these companies to provide context
            - Be specific about products, customers, markets, and competitive positioning

            {ticker1} Information:
            {ticker1_text}

            {ticker2} Information:
            {ticker2_text}

            Provide your analysis in this EXACT format:

            EXPLANATION:
            [2-3 sentences comparing their ACTUAL BUSINESS OPERATIONS in this dimension. Focus on what they do, not how they report it.]

            KEY SIMILARITIES:
            - [Concrete business similarity - what they both actually do]
            - [Concrete business similarity - shared characteristics]
            - [Concrete business similarity - common strategies or positions]

            KEY DIFFERENCES:
            - [Concrete business difference - how their operations differ]
            - [Concrete business difference - different approaches or markets]

            CONFIDENCE:
            [high/medium/low] - [Why, based on information quality and business knowledge]

            Remember: Compare BUSINESSES, not 10-K filing structures. Be substantive, not literal."""
        
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

def create_explainer(
    vector_store: Chroma,
    requests_per_minute: int = 8  # ADD THIS PARAMETER
) -> SimilarityExplainer:
    """
    Factory function to create a SimilarityExplainer.
    
    Args:
        vector_store: ChromaDB vector store
        requests_per_minute: Rate limit (8 is safe for free tier)
    
    Returns:
        Configured SimilarityExplainer instance
    """
    return SimilarityExplainer(
        vector_store=vector_store,
        requests_per_minute=requests_per_minute  # PASS IT THROUGH
    )