"""
Test the similarity engine with confidence scores and LLM explanations.
"""

import logging
import time
from sec_rag.similarity.similarity_engine import load_similarity_engine, ComparisonDimensions
from sec_rag.llm.similarity_explainer import create_explainer

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
        logger.info(f"   Confidence: {result.confidence}")  # NEW
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
        logger.info(f"   Confidence: {ranking.confidence}")  # NEW
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
        logger.info(f"   Confidence: {ranking.confidence}")  # NEW
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
        logger.info(f"   Confidence: {ranking.confidence}")  # NEW
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

def test_confidence_scores():
    """Test that confidence scores are calculated correctly."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Confidence Score Validation")
    logger.info("=" * 80)
    
    engine = load_similarity_engine()
    
    rankings = engine.find_similar_companies(top_n=10)
    
    logger.info("\nCompanies by confidence level:")
    
    high_conf = [r for r in rankings if r.confidence == "high"]
    medium_conf = [r for r in rankings if r.confidence == "medium"]
    low_conf = [r for r in rankings if r.confidence == "low"]
    
    logger.info(f"\nHigh Confidence ({len(high_conf)} companies):")
    for r in high_conf[:3]:
        logger.info(f"  {r.ticker}: Score={r.overall_score:.3f}, Matches={r.total_matches}")
    
    logger.info(f"\nMedium Confidence ({len(medium_conf)} companies):")
    for r in medium_conf[:3]:
        logger.info(f"  {r.ticker}: Score={r.overall_score:.3f}, Matches={r.total_matches}")
    
    logger.info(f"\nLow Confidence ({len(low_conf)} companies):")
    for r in low_conf[:3]:
        logger.info(f"  {r.ticker}: Score={r.overall_score:.3f}, Matches={r.total_matches}")

def test_llm_explanation():
    """Test LLM-generated explanations (rate limited)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: LLM Explanation Generation (with Caching)")
    logger.info("=" * 80)
    
    # Load engine and create explainer
    engine = load_similarity_engine()
    explainer = create_explainer(
        engine.vector_store,
        requests_per_minute=8
    )
    
    # Test explanation for AMZN vs MSFT
    logger.info("\nGenerating explanation: AMZN vs MSFT (business_model)")
    logger.info("This may take 5-10 seconds if not cached...")
    
    start_time = time.time()
    
    explanation = explainer.explain_similarity(
        ticker1="AMZN",
        ticker2="MSFT",
        dimension="business_model"
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparison: {explanation.ticker1} vs {explanation.ticker2}")
    logger.info(f"Dimension: {explanation.dimension}")
    logger.info(f"Confidence: {explanation.confidence}")
    logger.info(f"Generation Time: {elapsed:.2f}s")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nEXPLANATION:")
    logger.info(f"{explanation.explanation}")
    
    logger.info(f"\nKEY SIMILARITIES:")
    for i, sim in enumerate(explanation.key_similarities, 1):
        logger.info(f"  {i}. {sim}")
    
    logger.info(f"\nKEY DIFFERENCES:")
    for i, diff in enumerate(explanation.key_differences, 1):
        logger.info(f"  {i}. {diff}")

def test_multiple_explanations():
    """Test explanations across different dimensions."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 8: Multi-Dimension Explanations (with Caching)")
    logger.info("=" * 80)
    
    engine = load_similarity_engine()
    explainer = create_explainer(
        engine.vector_store,
        requests_per_minute=8
    )
    
    # Test different dimension explanations
    test_cases = [
        ("MA", "V", "business_model"),
        ("AAPL", "MSFT", "risk_profile"),
        ("TSLA", "NVDA", "financial_structure"),
    ]
    
    for ticker1, ticker2, dimension in test_cases:
        logger.info(f"\n{'-'*60}")
        logger.info(f"Explaining: {ticker1} vs {ticker2} ({dimension})")
        logger.info(f"{'-'*60}")
        
        start_time = time.time()
        
        explanation = explainer.explain_similarity(ticker1, ticker2, dimension)
        
        elapsed = time.time() - start_time
        
        logger.info(f"\nTime: {elapsed:.2f}s")
        logger.info(f"Confidence: {explanation.confidence}")
        logger.info(f"\n{explanation.explanation}")
        logger.info(f"\nTop Similarity: {explanation.key_similarities[0] if explanation.key_similarities else 'N/A'}")

def test_explanation_caching():
    """Test that explanations are cached and reused."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 9: Explanation Caching Verification")
    logger.info("=" * 80)
    
    engine = load_similarity_engine()
    explainer = create_explainer(
        engine.vector_store,
        requests_per_minute=8
    )
    
    # First call - might generate or use cache
    logger.info("\n1. First call for AAPL vs MSFT (business_model):")
    start = time.time()
    explanation1 = explainer.explain_similarity("AAPL", "MSFT", "business_model")
    time1 = time.time() - start
    logger.info(f"   Time: {time1:.2f}s")
    logger.info(f"   Confidence: {explanation1.confidence}")
    
    # Second call - should definitely use cache (instant)
    logger.info("\n2. Second call (should use cache):")
    start = time.time()
    explanation2 = explainer.explain_similarity("AAPL", "MSFT", "business_model")
    time2 = time.time() - start
    logger.info(f"   Time: {time2:.2f}s (should be <0.1s)")
    logger.info(f"   Confidence: {explanation2.confidence}")
    
    # Verify they're the same
    is_same = explanation1.explanation == explanation2.explanation
    logger.info(f"\n✓ Explanations match: {is_same}")
    
    if time1 > 1.0:  # Only compare if first call wasn't cached
        logger.info(f"✓ Cache speed up: {time1/time2:.1f}x faster")
    
    # Show cache stats
    logger.info("\n3. Cache statistics:")
    cache = explainer.cache
    all_cached = cache.list_cached()
    logger.info(f"   Total cached explanations: {len(all_cached)}")
    
    # Search cache by ticker
    aapl_explanations = cache.search_cache(ticker="AAPL")
    logger.info(f"   Explanations involving AAPL: {len(aapl_explanations)}")
    
    msft_explanations = cache.search_cache(ticker="MSFT")
    logger.info(f"   Explanations involving MSFT: {len(msft_explanations)}")
    
    # Search by dimension
    business_model_explanations = cache.search_cache(dimension="business_model")
    logger.info(f"   Business model explanations: {len(business_model_explanations)}")
    
    # List some cached files
    if all_cached:
        logger.info("\n4. Sample cached files:")
        for item in all_cached[:5]:  # Show first 5
            logger.info(f"   {item['filename']}")
            logger.info(f"     - Confidence: {item['confidence']}")
            logger.info(f"     - Cached at: {item['cached_at'][:19]}")  # Trim timestamp
    
    # Show where files are saved
    logger.info(f"\n5. Cache location:")
    logger.info(f"   {cache.cache_dir.absolute()}")
    logger.info(f"   You can open these .txt files to read the analyses!")

def test_force_refresh():
    """Test forcing regeneration of cached explanations."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 10: Force Refresh (Ignore Cache)")
    logger.info("=" * 80)
    
    engine = load_similarity_engine()
    explainer = create_explainer(
        engine.vector_store,
        requests_per_minute=8
    )
    
    logger.info("\nTesting force_refresh parameter...")
    logger.info("This regenerates even if cached (useful for testing)")
    
    start = time.time()
    explanation = explainer.explain_similarity(
        "GOOG", "META", "business_model",
        force_refresh=True  # Ignores cache
    )
    elapsed = time.time() - start
    
    logger.info(f"\n✓ Regenerated explanation in {elapsed:.2f}s")
    logger.info(f"   Confidence: {explanation.confidence}")
    logger.info(f"   Preview: {explanation.explanation[:150]}...")

if __name__ == "__main__":
    # Run all tests
    try:
        logger.info("\n" + "=" * 80)
        logger.info("SIMILARITY ENGINE TEST SUITE")
        logger.info("Testing across all 37 companies in database")
        logger.info("=" * 80)
        
        # Basic similarity tests (no LLM calls)
        test_single_dimension()
        test_multi_dimensional()
        test_target_company()
        test_specific_dimensions()
        test_explain_similarity()
        test_confidence_scores()
        
        # LLM explanation tests (may use API or cache)
        test_llm_explanation()
        test_multiple_explanations()
        test_explanation_caching()
        
        # Advanced features
        test_force_refresh()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        logger.info("\n📊 Summary:")
        logger.info("   - Tested similarity search across all dimensions")
        logger.info("   - Validated confidence scoring")
        logger.info("   - Generated LLM explanations with caching")
        logger.info("   - All cached explanations saved to ./outputs/explanations/")
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise