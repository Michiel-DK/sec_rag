#!/usr/bin/env python3
"""
Query 10-K Filings from Chroma Database

This script shows how to query your Chroma database with metadata filters
and use it with LangChain for Q&A.
"""

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def load_vectorstore(
    persist_directory: str = "./chroma_db",
    collection_name: str = "sec_filings"
) -> Chroma:
    """Load existing Chroma database"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore


def query_examples(vectorstore: Chroma):
    """Show various query examples with metadata filtering"""
    
    print("="*60)
    print("QUERY EXAMPLES")
    print("="*60)
    
    # Example 1: Simple similarity search
    print("\n[Example 1] Simple similarity search")
    print("-" * 60)
    query = "What are the main revenue sources?"
    results = vectorstore.similarity_search(query, k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Ticker: {result.metadata['ticker']}")
        print(f"  Date: {result.metadata['filing_date']}")
        print(f"  Preview: {result.page_content[:150]}...")
    
    # Example 2: Filter by specific ticker
    print("\n\n[Example 2] Search within specific company (AAPL)")
    print("-" * 60)
    query = "What are the risk factors?"
    results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"ticker": "AAPL"}
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Ticker: {result.metadata['ticker']}")
        print(f"  Date: {result.metadata['filing_date']}")
        print(f"  Preview: {result.page_content[:150]}...")
    
    # Example 3: Filter by year
    print("\n\n[Example 3] Search filings from specific year (2024)")
    print("-" * 60)
    query = "What is the revenue growth?"
    results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"fiscal_year": 2024}
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Ticker: {result.metadata['ticker']}")
        print(f"  Year: {result.metadata['fiscal_year']}")
        print(f"  Preview: {result.page_content[:150]}...")
    
    # Example 4: Multiple tickers (compare companies)
    print("\n\n[Example 4] Compare multiple companies")
    print("-" * 60)
    query = "What are the main business segments?"
    results = vectorstore.similarity_search(
        query,
        k=6,
        filter={"ticker": {"$in": ["AAPL", "MSFT", "GOOGL"]}}
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Ticker: {result.metadata['ticker']}")
        print(f"  Date: {result.metadata['filing_date']}")


def qa_with_sources(vectorstore: Chroma):
    """Q&A with source attribution"""
    
    print("\n\n" + "="*60)
    print("Q&A WITH LANGCHAIN")
    print("="*60)
    
    # Create custom prompt that includes metadata
    prompt_template = """Use the following pieces of context from SEC 10-K filings to answer the question. 
Each piece of context includes the company ticker and filing date.

Context:
{context}

Question: {question}

Provide a detailed answer and cite which companies and dates you're referencing.
Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    # Example questions
    questions = [
        "What are Apple's main revenue sources according to their latest 10-K?",
        "Compare the risk factors mentioned by different tech companies",
        "What are the main business segments described in the filings?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}] {question}")
        print("-" * 60)
        
        result = qa_chain.invoke({"query": question})
        
        print(f"\nAnswer:\n{result['result']}")
        
        print("\nSources:")
        for j, doc in enumerate(result['source_documents'][:3], 1):
            print(f"  {j}. {doc.metadata['ticker']} - {doc.metadata['filing_date']}")


def filtered_qa(vectorstore: Chroma):
    """Q&A with metadata filters"""
    
    print("\n\n" + "="*60)
    print("FILTERED Q&A")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Example: Ask about specific company
    print("\n[Query] Ask about AAPL specifically")
    print("-" * 60)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"ticker": "AAPL"}
            }
        ),
        return_source_documents=True
    )
    
    question = "What are the main products and services?"
    result = qa_chain.invoke({"query": question})
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['result']}")
    
    print("\nSources (all from AAPL):")
    for i, doc in enumerate(result['source_documents'][:3], 1):
        print(f"  {i}. {doc.metadata['ticker']} - {doc.metadata['filing_date']}")


def compare_companies(vectorstore: Chroma):
    """Compare specific companies"""
    
    print("\n\n" + "="*60)
    print("COMPANY COMPARISON")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    companies = ["AAPL", "MSFT"]
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {"ticker": {"$in": companies}}
            }
        ),
        return_source_documents=True
    )
    
    question = f"Compare {' and '.join(companies)}'s business models and revenue streams"
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    result = qa_chain.invoke({"query": question})
    print(f"\nAnswer:\n{result['result']}")
    
    print("\nSources:")
    for i, doc in enumerate(result['source_documents'][:5], 1):
        print(f"  {i}. {doc.metadata['ticker']} - {doc.metadata['filing_date']}")


def get_database_stats(vectorstore: Chroma):
    """Print database statistics"""
    
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    # Get all documents
    all_docs = vectorstore.get()
    
    # Count by ticker
    ticker_counts = {}
    year_counts = {}
    
    for metadata in all_docs['metadatas']:
        ticker = metadata.get('ticker', 'Unknown')
        year = metadata.get('fiscal_year', 'Unknown')
        
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        year_counts[year] = year_counts.get(year, 0) + 1
    
    print(f"\nTotal chunks: {len(all_docs['ids'])}")
    print(f"Unique tickers: {len(ticker_counts)}")
    
    print("\nChunks per ticker:")
    for ticker, count in sorted(ticker_counts.items()):
        print(f"  {ticker}: {count}")
    
    print("\nChunks per year:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count}")


def main():
    """Main execution"""
    
    print("="*60)
    print("10-K Filings Query Interface")
    print("="*60)
    
    # Load vectorstore
    print("\nLoading Chroma database...")
    vectorstore = load_vectorstore()
    
    # Show database stats
    get_database_stats(vectorstore)
    
    # Run examples
    query_examples(vectorstore)
    
    # Q&A examples
    qa_with_sources(vectorstore)
    
    # Filtered queries
    filtered_qa(vectorstore)
    
    # Company comparison
    compare_companies(vectorstore)
    
    print("\n" + "="*60)
    print("✓ All examples complete!")
    print("="*60)


if __name__ == "__main__":
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    else:
        main()