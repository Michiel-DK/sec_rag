# test_small_batch.py

import os
import logging
from pathlib import Path
from sec_rag.chroma.load_filings_to_chroma import (
    load_10k_filings,
    chunk_documents,
    create_vector_store_with_retry,
    FILINGS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_small_batch():
    """
    Test the pipeline on just 3 companies: AMZN, V, NFLX
    """
    logger.info("=" * 80)
    logger.info("SMALL BATCH TEST - 3 Companies")
    logger.info("=" * 80)
    
    # Load all documents
    all_documents = load_10k_filings(FILINGS_DIR)
    
    # Filter to just 3 companies
    TEST_TICKERS = ['AMZN', 'V', 'NFLX']
    test_documents = [
        doc for doc in all_documents 
        if doc.metadata.get('ticker') in TEST_TICKERS
    ]
    
    logger.info(f"\n✓ Filtered to {len(test_documents)} test documents")
    logger.info(f"  Tickers: {', '.join(TEST_TICKERS)}")
    
    # Chunk documents
    logger.info("\nChunking documents...")
    chunks = chunk_documents(test_documents)
    logger.info(f"✓ Created {len(chunks)} chunks")
    
    # Show chunk distribution
    for ticker in TEST_TICKERS:
        ticker_chunks = [c for c in chunks if c.metadata.get('ticker') == ticker]
        logger.info(f"  {ticker}: {len(ticker_chunks)} chunks")
    
    # Create vector store with small batch
    logger.info("\nCreating test vector store...")
    vectorstore = create_vector_store_with_retry(
        chunks,
        persist_dir="./chroma_db_test",  # Different directory for testing
        batch_size=50,  # Smaller batches for testing
        max_retries=3
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    
    # Test a query
    logger.info("\nTesting a sample query...")
    results = vectorstore.similarity_search(
        "What are Amazon's main business segments?",
        k=3,
        filter={"ticker": "AMZN"}
    )
    
    logger.info(f"✓ Query returned {len(results)} results")
    
    if results:
        logger.info("\nSample result:")
        logger.info(f"  Ticker: {results[0].metadata.get('ticker')}")
        logger.info(f"  Section: {results[0].metadata.get('section')}")
        logger.info(f"  Text preview: {results[0].page_content[:200]}...")
    else:
        logger.warning("⚠️  No results returned - embeddings may have failed")
    
    return vectorstore

if __name__ == "__main__":
    test_small_batch()