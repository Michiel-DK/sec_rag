from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# test_queries.py - UPDATED WITH DEBUG INFO

def test_validation_queries():
    """Test retrieval with debugging."""
    print("=" * 80)
    print("QUERY VALIDATION TEST")
    print("=" * 80)
    
    # Load the vector store
    print("\nLoading vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="sec_filings"
    )
    
    # CHECK: How many documents in the store?
    collection_count = vectorstore._collection.count()
    print(f"✓ Vector store loaded")
    print(f"✓ Total documents in store: {collection_count}")
    
    if collection_count == 0:
        print("\n⚠️  ERROR: Vector store is EMPTY!")
        print("Run: python -m sec_rag.chroma.load_filings_to_chroma first")
        return
    
    # Test a simple query first
    print("\n" + "=" * 80)
    print("TEST QUERY: Simple retrieval test")
    print("-" * 80)
    
    try:
        test_results = vectorstore.similarity_search("revenue", k=3)
        print(f"✓ Found {len(test_results)} results for 'revenue'")
        
        if test_results:
            print("\nSample result:")
            print(f"  Ticker: {test_results[0].metadata.get('ticker')}")
            print(f"  Section: {test_results[0].metadata.get('section')}")
            print(f"  Text: {test_results[0].page_content[:200]}...")
        else:
            print("⚠️  No results found - embeddings may not be working")
            
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return
    
    # Now run the actual test queries
    test_queries = [
        ("AMZN", "What are Amazon's main business segments?"),
        ("V", "Who are Visa's directors?"),
        ("NFLX", "Explain Netflix's content strategy"),
    ]
    
    for ticker, query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Ticker: {ticker}")
        print(f"Query: {query}")
        print(f"{'-' * 80}")
        
        try:
            # Try with ticker filter
            results = vectorstore.similarity_search(
                query,
                k=3,
                filter={"ticker": ticker}
            )
            
            print(f"Found {len(results)} results")
            
            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Ticker: {doc.metadata.get('ticker')}")
                print(f"  Section: {doc.metadata.get('section')}")
                print(f"  Fiscal Year: {doc.metadata.get('fiscal_year')}")
                print(f"  Text: {doc.page_content[:300]}...")
                
        except Exception as e:
            print(f"✗ Query failed: {e}")

if __name__ == "__main__":
    test_validation_queries()