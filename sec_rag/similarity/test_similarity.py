"""
Test the similarity engine with real queries.
"""

import logging
from sec_rag.similarity.similarity_engine import load_similarity_engine, ComparisonDimensions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_dimension():
    """Test finding similarities in one dimension."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Single Dimension Search (Business Model)")
    logger.info("=" * 80)
    
    # Load engine
    engine = load_similarity_engine()
    
    # Search business model dimension
    dimension = ComparisonDimensions.business_model()
    results = engine.find_similar_by_dimension(dimension)
    
    # Display results
    logger.info(f"\nTop 10 companies by business model similarity:")
    for i, result in enumerate(results[:10], 1):
        logger.info(f"\n{i}. {result.ticker}")
        logger.info(f"   Score: {result.score:.2f}")
        logger.info(f"   Matches: {result.match_count}")
        logger.info(f"   Sample: {result.sample_chunks[0][:100]}...")

def test_multi_dimensional():
    """Test finding similarities across all dimensions."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Multi-Dimensional Comparison")
    logger.info("=" * 80)
    
    # Load engine
    engine = load_similarity_engine()
    
    # Find similar companies across all dimensions
    rankings = engine.find_similar_companies(top_n=10)
    
    # Display results
    logger.info(f"\nTop 10 most similar companies (overall):")
    for i, ranking in enumerate(rankings, 1):
        logger.info(f"\n{i}. {ranking.ticker}")
        logger.info(f"   Overall Score: {ranking.overall_score:.3f}")
        logger.info(f"   Total Matches: {ranking.total_matches}")
        logger.info(f"   Dimension Breakdown:")
        for dim, score in ranking.dimension_scores.items():
            logger.info(f"     {dim}: {score:.3f}")

def test_target_company():
    """Test finding companies similar to a specific target."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Find Companies Similar to AMZN")
    logger.info("=" * 80)
    
    # Load engine
    engine = load_similarity_engine()
    
    # Find companies similar to Amazon
    rankings = engine.find_similar_companies(
        target_ticker="AMZN",
        top_n=5
    )
    
    # Display results
    logger.info(f"\nTop 5 companies most similar to AMZN:")
    for i, ranking in enumerate(rankings, 1):
        logger.info(f"\n{i}. {ranking.ticker}")
        logger.info(f"   Overall Score: {ranking.overall_score:.3f}")
        logger.info(f"   Dimension Breakdown:")
        for dim, score in sorted(
            ranking.dimension_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"     {dim}: {score:.3f}")

def test_specific_dimensions():
    """Test comparing companies on specific dimensions only."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Compare on Risk Profile & Financial Structure Only")
    logger.info("=" * 80)
    
    # Load engine
    engine = load_similarity_engine()
    
    # Compare only on risk and financial dimensions
    rankings = engine.find_similar_companies(
        dimensions=["risk_profile", "financial_structure"],
        top_n=10
    )
    
    # Display results
    logger.info(f"\nTop 10 companies by risk/financial similarity:")
    for i, ranking in enumerate(rankings, 1):
        logger.info(f"\n{i}. {ranking.ticker}")
        logger.info(f"   Overall Score: {ranking.overall_score:.3f}")
        logger.info(f"   Risk Profile: {ranking.dimension_scores.get('risk_profile', 0):.3f}")
        logger.info(f"   Financial: {ranking.dimension_scores.get('financial_structure', 0):.3f}")

def test_explain_similarity():
    """Test explaining why two companies are similar."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Explain Similarity Between AMZN and WMT")
    logger.info("=" * 80)
    
    # Load engine
    engine = load_similarity_engine()
    
    # Explain business model similarity
    explanation = engine.explain_similarity(
        ticker1="AMZN",
        ticker2="WMT",
        dimension="business_model"
    )
    
    logger.info(f"\nWhy are AMZN and WMT similar in business model?")
    logger.info(f"\nAMZN relevant chunks:")
    for i, chunk in enumerate(explanation['AMZN_chunks'][:3], 1):
        logger.info(f"\n  {i}. {chunk[:300]}...")
    
    logger.info(f"\nWMT relevant chunks:")
    for i, chunk in enumerate(explanation['WMT_chunks'][:3], 1):
        logger.info(f"\n  {i}. {chunk[:300]}...")

if __name__ == "__main__":
    # Run all tests
    try:
        test_single_dimension()
        test_multi_dimensional()
        test_target_company()
        test_specific_dimensions()
        test_explain_similarity()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL TESTS COMPLETED")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        raise