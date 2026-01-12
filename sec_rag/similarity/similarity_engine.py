"""
Core engine for finding similar companies using RAG.
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .dimension_queries import ComparisonDimensions, DimensionQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompanySimilarity:
    """Result of company similarity comparison."""
    ticker: str
    dimension: str
    score: float
    match_count: int
    confidence: str  # NEW: "high", "medium", "low"
    sample_chunks: List[str]

@dataclass
class CompanyRanking:
    """Overall ranking of a company across all dimensions."""
    ticker: str
    overall_score: float
    dimension_scores: Dict[str, float]
    total_matches: int
    confidence: str  # NEW: "high", "medium", "low"str, float]
    total_matches: int
    
def calculate_confidence(score: float, match_count: int, dimension: str = None) -> str:
    """
    Calculate confidence level for similarity scores.
    
    Args:
        score: Normalized similarity score (0-1 for overall, higher for dimensions)
        match_count: Number of matching chunks found
        dimension: Optional dimension name for dimension-specific thresholds
    
    Returns:
        Confidence level: "high", "medium", or "low"
    
    Confidence Criteria:
    - HIGH: Strong evidence with many matches
    - MEDIUM: Moderate evidence with some matches  
    - LOW: Weak evidence with few matches
    """
    if dimension:
        # Dimension-specific thresholds
        # Dimensions have scores > 1, so use higher thresholds
        if score >= 8.0 and match_count >= 8:
            return "high"
        elif score >= 4.0 and match_count >= 4:
            return "medium"
        else:
            return "low"
    else:
        # Overall score thresholds (scores are 0-1 range)
        if score >= 0.7 and match_count >= 20:
            return "high"
        elif score >= 0.4 and match_count >= 10:
            return "medium"
        else:
            return "low"


# Update the find_similar_by_dimension method to include confidence:

def find_similar_by_dimension(
    self,
    dimension: DimensionQuery,
    exclude_tickers: Optional[List[str]] = None
) -> List[CompanySimilarity]:
    """
    Find similar companies for a specific dimension.
    NOW WITH CONFIDENCE SCORES!
    """
    exclude_tickers = exclude_tickers or []
    
    logger.info(f"Searching dimension: {dimension.dimension}")
    logger.info(f"  Queries: {len(dimension.queries)}")
    logger.info(f"  Sections: {dimension.sections}")
    
    # Aggregate results across all queries in this dimension
    company_matches = defaultdict(lambda: {
        'score': 0.0,
        'count': 0,
        'chunks': []
    })
    
    for query_text in dimension.queries:
        logger.debug(f"  Query: {query_text[:60]}...")
        
        results = self._search_with_fallback(query_text, dimension.sections)
        
        for doc in results:
            ticker = doc.metadata.get('ticker', 'Unknown')
            
            if ticker in exclude_tickers or ticker == 'Unknown':
                continue
            
            company_matches[ticker]['score'] += 1.0
            company_matches[ticker]['count'] += 1
            company_matches[ticker]['chunks'].append(doc.page_content[:200])
    
    # Convert to CompanySimilarity objects WITH CONFIDENCE
    similarities = []
    for ticker, data in company_matches.items():
        if data['count'] >= self.min_matches:
            # Calculate confidence for this dimension score
            confidence = calculate_confidence(
                score=data['score'],
                match_count=data['count'],
                dimension=dimension.dimension
            )
            
            similarities.append(CompanySimilarity(
                ticker=ticker,
                dimension=dimension.dimension,
                score=data['score'],
                match_count=data['count'],
                confidence=confidence,  # NEW
                sample_chunks=data['chunks'][:3]
            ))
    
    similarities.sort(key=lambda x: x.score, reverse=True)
    
    logger.info(f"  Found {len(similarities)} companies with >= {self.min_matches} matches")
    
    return similarities

class SimilarityEngine:
    """
    Engine for finding similar companies using multi-query RAG.
    """
    
    def __init__(
        self,
        vector_store: Chroma,
        top_k_per_query: int = 10,
        min_matches: int = 2
    ):
        """
        Initialize the similarity engine.
        
        Args:
            vector_store: ChromaDB vector store with embedded 10-Ks
            top_k_per_query: How many chunks to retrieve per query
            min_matches: Minimum matches required to include a company
        """
        self.vector_store = vector_store
        self.top_k_per_query = top_k_per_query
        self.min_matches = min_matches
    
    def find_similar_by_dimension(
        self,
        dimension: DimensionQuery,
        exclude_tickers: Optional[List[str]] = None
    ) -> List[CompanySimilarity]:
        """
        Find similar companies for a specific dimension.
        NOW WITH CONFIDENCE SCORES!
        """
        exclude_tickers = exclude_tickers or []
        
        logger.info(f"Searching dimension: {dimension.dimension}")
        logger.info(f"  Queries: {len(dimension.queries)}")
        logger.info(f"  Sections: {dimension.sections}")
        
        # Aggregate results across all queries in this dimension
        company_matches = defaultdict(lambda: {
            'score': 0.0,
            'count': 0,
            'chunks': []
        })
        
        for query_text in dimension.queries:
            logger.debug(f"  Query: {query_text[:60]}...")
            
            results = self._search_with_fallback(query_text, dimension.sections)
            
            for doc in results:
                ticker = doc.metadata.get('ticker', 'Unknown')
                
                if ticker in exclude_tickers or ticker == 'Unknown':
                    continue
                
                company_matches[ticker]['score'] += 1.0
                company_matches[ticker]['count'] += 1
                company_matches[ticker]['chunks'].append(doc.page_content[:200])
        
        # Convert to CompanySimilarity objects WITH CONFIDENCE
        similarities = []
        for ticker, data in company_matches.items():
            if data['count'] >= self.min_matches:
                # Calculate confidence for this dimension score
                confidence = calculate_confidence(
                    score=data['score'],
                    match_count=data['count'],
                    dimension=dimension.dimension
                )
                
                similarities.append(CompanySimilarity(
                    ticker=ticker,
                    dimension=dimension.dimension,
                    score=data['score'],
                    match_count=data['count'],
                    confidence=confidence,  # NEW
                    sample_chunks=data['chunks'][:3]
                ))
        
        similarities.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"  Found {len(similarities)} companies with >= {self.min_matches} matches")
        
        return similarities
    
    def _search_with_fallback(
        self,
        query: str,
        sections: List[str]
    ) -> List:
        """
        Search with section filtering, fallback to no filter if needed.
        """
        # Try with section filter first
        for section in sections:
            try:
                results = self.vector_store.similarity_search(
                    query,
                    k=self.top_k_per_query,
                    filter={"section": section}
                )
                
                if results:
                    return results
            except Exception as e:
                logger.debug(f"Section filter failed for {section}: {e}")
        
        # Fallback: search without section filter
        try:
            results = self.vector_store.similarity_search(
                query,
                k=self.top_k_per_query
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def find_similar_companies(
        self,
        target_ticker: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[CompanyRanking]:
        """
        Find companies similar across multiple dimensions.
        NOW WITH CONFIDENCE SCORES AND PROPER DIMENSION FILTERING!
        """
        # Get dimensions to compare - WITH PROPER VALIDATION
        if dimensions is None:
            dimension_queries = ComparisonDimensions.all_dimensions()
            logger.info("No dimensions specified - using all 5 dimensions")
        else:
            # FIX: Normalize dimension names and validate
            normalized_dims = [d.lower().strip().replace(' ', '_') for d in dimensions]
            dimension_queries = []
            
            valid_dimensions = {
                'business_model', 'risk_profile', 'financial_structure',
                'geographic_footprint', 'legal_matters'
            }
            
            for dim in normalized_dims:
                if dim not in valid_dimensions:
                    logger.warning(f"Invalid dimension '{dim}', skipping. Valid: {valid_dimensions}")
                    continue
                
                try:
                    dim_query = ComparisonDimensions.get_dimension(dim)
                    dimension_queries.append(dim_query)
                    logger.debug(f"Added dimension: {dim}")
                except ValueError as e:
                    logger.warning(f"Could not load dimension '{dim}': {e}")
            
            if not dimension_queries:
                logger.warning("No valid dimensions provided, falling back to all dimensions")
                dimension_queries = ComparisonDimensions.all_dimensions()
            else:
                logger.info(f"✓ Filtered to {len(dimension_queries)} dimension(s): {[d.dimension for d in dimension_queries]}")
        
        logger.info("=" * 80)
        logger.info(f"Finding similar companies")
        logger.info(f"  Target: {target_ticker or 'All companies'}")
        logger.info(f"  Active Dimensions ({len(dimension_queries)}): {[d.dimension for d in dimension_queries]}")
        logger.info("=" * 80)
        
        exclude = [target_ticker] if target_ticker else []
        
        # Search each dimension
        all_results = {}
        for dim_query in dimension_queries:
            logger.info(f"Searching dimension: {dim_query.dimension}")
            similarities = self.find_similar_by_dimension(dim_query, exclude)
            all_results[dim_query.dimension] = {
                'results': similarities,
                'weight': dim_query.weight
            }
            logger.info(f"  ✓ Found {len(similarities)} matches in {dim_query.dimension}")
        
        # Aggregate scores across dimensions
        company_scores = defaultdict(lambda: {
            'weighted_score': 0.0,
            'total_matches': 0,
            'dimension_scores': {}
        })
        
        for dimension, data in all_results.items():
            weight = data['weight']
            
            for similarity in data['results']:
                ticker = similarity.ticker
                
                max_score = max([s.score for s in data['results']], default=1.0)
                normalized_score = similarity.score / max_score if max_score > 0 else 0
                
                company_scores[ticker]['weighted_score'] += normalized_score * weight
                company_scores[ticker]['total_matches'] += similarity.match_count
                company_scores[ticker]['dimension_scores'][dimension] = normalized_score
        
        # Convert to CompanyRanking objects WITH CONFIDENCE
        rankings = []
        for ticker, data in company_scores.items():
            # Calculate overall confidence
            confidence = calculate_confidence(
                score=data['weighted_score'],
                match_count=data['total_matches']
            )
            
            rankings.append(CompanyRanking(
                ticker=ticker,
                overall_score=data['weighted_score'],
                dimension_scores=data['dimension_scores'],
                total_matches=data['total_matches'],
                confidence=confidence
            ))
        
        rankings.sort(key=lambda x: x.overall_score, reverse=True)
        
        logger.info(f"\n✓ Returning top {min(top_n, len(rankings))} similar companies")
        
        return rankings[:top_n]

    
    def explain_similarity(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str
    ) -> Dict:
        """
        Explain why two companies are similar in a specific dimension.
        
        Returns relevant chunks from both companies.
        """
        dim_query = ComparisonDimensions.get_dimension(dimension)
        
        # Get chunks for both companies
        results = {}
        for ticker in [ticker1, ticker2]:
            ticker_chunks = []
            
            for query_text in dim_query.queries:
                chunks = self.vector_store.similarity_search(
                    query_text,
                    k=3,
                    filter={"ticker": ticker}
                )
                ticker_chunks.extend(chunks)
            
            results[ticker] = ticker_chunks
        
        return {
            'dimension': dimension,
            'ticker1': ticker1,
            'ticker2': ticker2,
            f'{ticker1}_chunks': [c.page_content for c in results[ticker1][:5]],
            f'{ticker2}_chunks': [c.page_content for c in results[ticker2][:5]]
        }

def load_similarity_engine(
    persist_dir: str = "./chroma_db",
    top_k_per_query: int = 10
) -> SimilarityEngine:
    """
    Load the similarity engine with existing vector store.
    """
    logger.info(f"Loading vector store from {persist_dir}...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="sec_filings"
    )
    
    count = vector_store._collection.count()
    logger.info(f"✓ Loaded vector store with {count} chunks")
    
    if count == 0:
        raise ValueError(
            "Vector store is empty! Run load_filings_to_chroma.py first."
        )
    
    return SimilarityEngine(
        vector_store=vector_store,
        top_k_per_query=top_k_per_query
    )