import os
from sec_rag.chroma.load_filings_to_chroma import (
    load_10k_filings,  # Changed from load_and_process_filings
    chunk_documents,
    FILINGS_DIR
)

def test_validation_queries():
    """
    Test the chunking strategy with validation queries.
    """
    print("=" * 80)
    print("CHUNKING VALIDATION TEST")
    print("=" * 80)
    
    try:
        # Load documents
        print("\n1. Loading documents...")
        documents = load_10k_filings(FILINGS_DIR)
        print(f"   ✓ Loaded {len(documents)} document(s)")
        
        # Show which companies were loaded
        tickers = set(doc.metadata.get('ticker', 'Unknown') for doc in documents)
        print(f"   ✓ Tickers: {', '.join(sorted(tickers))}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nPlease ensure:")
        print(f"  1. Directory exists: {FILINGS_DIR}")
        print(f"  2. Structure: {FILINGS_DIR}/TICKER/TICKER_10-K_DATE_ACCESSION.txt")
        return
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Chunk documents
    print("\n2. Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # Analyze chunks by ticker
    print("\n3. Chunk distribution by ticker:")
    ticker_counts = {}
    for chunk in chunks:
        ticker = chunk.metadata.get('ticker', 'Unknown')
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    
    for ticker, count in sorted(ticker_counts.items()):
        print(f"   {ticker}: {count} chunks")
    
    # Analyze chunks by fiscal year
    print("\n4. Chunk distribution by fiscal year:")
    year_counts = {}
    for chunk in chunks:
        year = chunk.metadata.get('fiscal_year', 'Unknown')
        year_counts[year] = year_counts.get(year, 0) + 1
    
    for year, count in sorted(year_counts.items()):
        print(f"   {year}: {count} chunks")
    
    # Analyze chunks by section
    print("\n5. Chunk distribution by section (top 10):")
    section_counts = {}
    for chunk in chunks:
        section = chunk.metadata.get('section', 'Unknown')
        section_counts[section] = section_counts.get(section, 0) + 1
    
    sorted_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
    for section, count in sorted_sections[:10]:
        print(f"   {section}: {count} chunks")
    
    # Find director/officer chunks
    print("\n6. Director/Officer information chunks:")
    director_chunks = [c for c in chunks if c.metadata.get('contains_director_info', False)]
    print(f"   Found {len(director_chunks)} chunks with director/officer info")
    
    if director_chunks:
        # Show distribution by ticker
        dir_by_ticker = {}
        for chunk in director_chunks:
            ticker = chunk.metadata.get('ticker', 'Unknown')
            dir_by_ticker[ticker] = dir_by_ticker.get(ticker, 0) + 1
        
        print("   Distribution by ticker:")
        for ticker, count in sorted(dir_by_ticker.items()):
            print(f"     {ticker}: {count} chunks")
        
        # Show sample
        sample = director_chunks[0]
        print(f"\n   Sample chunk:")
        print(f"     Ticker: {sample.metadata.get('ticker')}")
        print(f"     Section: {sample.metadata.get('section')}")
        print(f"     Text preview: {sample.page_content[:200]}...")
    
    # Find legal matter chunks
    print("\n7. Legal matter chunks:")
    legal_chunks = [c for c in chunks if c.metadata.get('is_legal_matter', False)]
    print(f"   Found {len(legal_chunks)} chunks with legal matters")
    
    if legal_chunks:
        legal_by_ticker = {}
        for chunk in legal_chunks:
            ticker = chunk.metadata.get('ticker', 'Unknown')
            legal_by_ticker[ticker] = legal_by_ticker.get(ticker, 0) + 1
        
        print("   Distribution by ticker:")
        for ticker, count in sorted(legal_by_ticker.items()):
            print(f"     {ticker}: {count} chunks")
    
    # Find chunks with tables
    print("\n8. Chunks containing tables:")
    table_chunks = [c for c in chunks if c.metadata.get('contains_table', False)]
    print(f"   Found {len(table_chunks)} chunks with tables")
    print(f"   Percentage: {len(table_chunks)/len(chunks)*100:.1f}%")
    
    # Find financial note chunks
    print("\n9. Financial statement notes:")
    note_chunks = [c for c in chunks if 'note_number' in c.metadata]
    print(f"   Found {len(note_chunks)} chunks in financial notes")
    
    # Group by note number
    note_distribution = {}
    for chunk in note_chunks:
        note = chunk.metadata.get('note_number', 'Unknown')
        note_distribution[note] = note_distribution.get(note, 0) + 1
    
    if note_distribution:
        print("   Top 5 notes by chunk count:")
        sorted_notes = sorted(note_distribution.items(), key=lambda x: x[1], reverse=True)
        for note, count in sorted_notes[:5]:
            print(f"     {note}: {count} chunks")
    
    # Chunk size analysis
    print("\n10. Chunk size analysis:")
    chunk_sizes = [len(c.page_content) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)
    
    print(f"    Average chunk size: {avg_size:.0f} characters")
    print(f"    Min chunk size: {min_size} characters")
    print(f"    Max chunk size: {max_size} characters")
    
    # Geographic region analysis
    print("\n11. Geographic region distribution:")
    region_chunks = [c for c in chunks if 'geographic_region' in c.metadata]
    print(f"     Found {len(region_chunks)} chunks with geographic region tags")
    
    if region_chunks:
        region_counts = {}
        for chunk in region_chunks:
            region = chunk.metadata.get('geographic_region', 'Unknown')
            region_counts[region] = region_counts.get(region, 0) + 1
        
        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     {region}: {count} chunks")
    
    # Risk factors analysis
    print("\n12. Risk factor categorization:")
    risk_chunks = [c for c in chunks if 'risk_category' in c.metadata]
    print(f"     Found {len(risk_chunks)} chunks with risk categories")
    
    if risk_chunks:
        risk_counts = {}
        for chunk in risk_chunks:
            risk = chunk.metadata.get('risk_category', 'Unknown')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print("     Top risk categories:")
        sorted_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
        for risk, count in sorted_risks[:5]:
            print(f"       {risk}: {count} chunks")
    
    # Metadata completeness check
    print("\n13. Metadata completeness:")
    required_fields = ['ticker', 'fiscal_year', 'section', 'filing_type']
    for field in required_fields:
        missing = sum(1 for c in chunks if field not in c.metadata or c.metadata[field] == 'Unknown')
        if missing > 0:
            print(f"     ⚠ {field}: {missing} chunks missing ({missing/len(chunks)*100:.1f}%)")
        else:
            print(f"     ✓ {field}: Complete")
    
    print("\n" + "=" * 80)
    print("VALIDATION TEST COMPLETE")
    print("=" * 80)
    
    # Summary statistics
    print("\n📊 SUMMARY:")
    print(f"   Total documents: {len(documents)}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Companies: {len(ticker_counts)}")
    print(f"   Average chunks per document: {len(chunks)/len(documents):.0f}")

if __name__ == "__main__":
    test_validation_queries()