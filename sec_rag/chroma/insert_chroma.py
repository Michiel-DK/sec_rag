#!/usr/bin/env python3
"""
Load 10-K Filings into Chroma Vector Database

This script:
1. Reads all 10-K filings from your directory structure
2. Extracts metadata from folder names and filenames
3. Chunks the documents
4. Stores them in Chroma with rich metadata
"""

from pathlib import Path
import re
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm

def extract_metadata_from_path(file_path: Path) -> Dict[str, str]:
    """
    Extract metadata from file path structure.
    
    Expected structure: 10k_filings/TICKER/TICKER_10-K_DATE_ACCESSION.txt
    Example: 10k_filings/AAPL/AAPL_10-K_2025-10-31_0000320193.txt
    """
    # Get ticker from parent directory name
    ticker = file_path.parent.name
    
    # Parse filename: TICKER_10-K_DATE_ACCESSION.txt
    filename = file_path.stem  # Remove .txt extension
    parts = filename.split('_')
    
    metadata = {
        "ticker": ticker,
        "filing_type": "10-K",
        "source_file": file_path.name,
        "file_path": str(file_path)
    }
    
    # Extract date (format: YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_match = re.search(date_pattern, filename)
    if date_match:
        filing_date = date_match.group()
        metadata["filing_date"] = filing_date
        metadata["fiscal_year"] = int(filing_date.split('-')[0])
    
    # Extract accession number (last part before extension)
    if len(parts) >= 4:
        metadata["accession_number"] = parts[-1]
    
    return metadata


def load_10k_filings(
    filings_dir: str = "10k_filings",
    pattern: str = "**/*.txt"
) -> List[Document]:
    """
    Load all 10-K filings from directory structure.
    
    Args:
        filings_dir: Root directory containing ticker subdirectories
        pattern: Glob pattern for finding filing files
    
    Returns:
        List of LangChain Documents with metadata
    """
    filings_path = Path(filings_dir)
    
    if not filings_path.exists():
        raise FileNotFoundError(f"Directory not found: {filings_dir}")
    
    # Find all .txt files
    filing_files = list(filings_path.glob(pattern))
    
    if not filing_files:
        raise FileNotFoundError(f"No .txt files found in {filings_dir}")
    
    print(f"Found {len(filing_files)} filing files")
    
    documents = []
    
    for file_path in tqdm(filing_files, desc="Loading filings"):
        try:
            # Extract metadata from path
            metadata = extract_metadata_from_path(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks while preserving metadata.
    
    Args:
        documents: List of full documents
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked documents with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    print(f"Chunking {len(documents)} documents...")
    chunked_docs = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        # Split the document
        chunks = text_splitter.split_text(doc.page_content)
        
        # Create new documents for each chunk with original metadata
        for i, chunk in enumerate(chunks):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            chunked_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            chunked_docs.append(chunked_doc)
    
    print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs


def create_chroma_db(
    documents: List[Document],
    collection_name: str = "sec_10k_filings",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "text-embedding-004",
    batch_size: int = 100
) -> Chroma:
    """
    Create Chroma vector database from documents.
    
    Args:
        documents: List of chunked documents
        collection_name: Name for the Chroma collection
        persist_directory: Where to save the database
        embedding_model: OpenAI embedding model to use
        batch_size: Number of documents to process at once
    
    Returns:
        Chroma vectorstore instance
    """
    print(f"\nCreating Chroma database...")
    print(f"Collection: {collection_name}")
    print(f"Persist directory: {persist_directory}")
    print(f"Embedding model: {embedding_model}")
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Create Chroma vectorstore
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Add documents in batches
    print(f"\nAdding {len(documents)} documents in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Adding to Chroma"):
        batch = documents[i:i+batch_size]
        vectorstore.add_documents(batch)
    
    print(f"\n✓ Successfully created Chroma database with {len(documents)} chunks")
    
    return vectorstore


def main():
    """Main execution function"""
    
    # Configuration
    FILINGS_DIR = "10k_filings"          # Your filings directory
    PERSIST_DIR = "./chroma_db"          # Where to save Chroma DB
    COLLECTION_NAME = "sec_10k_filings"  # Collection name
    CHUNK_SIZE = 1000                    # Characters per chunk
    CHUNK_OVERLAP = 200                  # Overlap between chunks
    BATCH_SIZE = 100                     # Batch size for adding to Chroma
    
    print("="*60)
    print("10-K Filings to Chroma Vector Database")
    print("="*60)
    
    # Step 1: Load filings
    print("\n[Step 1/3] Loading 10-K filings from disk...")
    documents = load_10k_filings(filings_dir=FILINGS_DIR)
    
    # Show sample metadata
    if documents:
        print("\nSample metadata from first document:")
        for key, value in documents[0].metadata.items():
            print(f"  {key}: {value}")
    
    # Step 2: Chunk documents
    print(f"\n[Step 2/3] Chunking documents...")
    chunked_documents = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Step 3: Create Chroma database
    print(f"\n[Step 3/3] Creating Chroma vector database...")
    vectorstore = create_chroma_db(
        documents=chunked_documents,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Test query
    print("\n" + "="*60)
    print("Testing the database...")
    print("="*60)
    
    test_query = "What are the main business segments?"
    print(f"\nTest query: '{test_query}'")
    
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Ticker: {result.metadata.get('ticker')}")
        print(f"Date: {result.metadata.get('filing_date')}")
        print(f"Chunk: {result.metadata.get('chunk_index')}/{result.metadata.get('total_chunks')}")
        print(f"Content preview: {result.page_content[:200]}...")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)
    
    # Count documents per ticker
    all_docs = vectorstore.get()
    tickers = {}
    for metadata in all_docs['metadatas']:
        ticker = metadata.get('ticker', 'Unknown')
        tickers[ticker] = tickers.get(ticker, 0) + 1
    
    print(f"\nTotal chunks: {len(all_docs['ids'])}")
    print(f"Unique tickers: {len(tickers)}")
    print("\nChunks per ticker:")
    for ticker, count in sorted(tickers.items()):
        print(f"  {ticker}: {count}")
    
    print(f"\nDatabase saved to: {Path(PERSIST_DIR).absolute()}")
    print("\n✓ All done!")
    
    return vectorstore


if __name__ == "__main__":
    # Make sure you have OPENAI_API_KEY set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    else:
        vectorstore = main()