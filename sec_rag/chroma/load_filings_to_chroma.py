import os
import logging
import re
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
FILINGS_DIR = "./10k_filings"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

def extract_metadata_from_path(file_path: Path) -> Dict[str, Any]:
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
        "ticker": ticker.upper(),
        "company": ticker.upper(),  # Alias for consistency with old code
        "filing_type": "10-K",
        "form_type": "10-K",  # Alias for consistency with old code
        "source_file": file_path.name,
        "file_path": str(file_path),
        "source": str(file_path),  # For compatibility
        "filename": filename
    }
    
    # Extract date (format: YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_match = re.search(date_pattern, filename)
    if date_match:
        filing_date = date_match.group()
        metadata["filing_date"] = filing_date
        metadata["fiscal_year"] = int(filing_date.split('-')[0])
    else:
        logger.warning(f"Could not extract date from filename: {filename}")
        metadata["fiscal_year"] = "unknown"
    
    # Extract accession number (last part before extension)
    if len(parts) >= 4:
        metadata["accession_number"] = parts[-1]
    else:
        logger.warning(f"Could not extract accession number from filename: {filename}")
    
    return metadata

def detect_section(text: str) -> str:
    """Detect which SEC section this chunk belongs to - IMPROVED VERSION."""
    text_upper = text.upper()
    
    # More flexible pattern matching
    patterns = {
        'Item 1A - Risk Factors': [
            r'ITEM\s*1A[\.\s]',
            r'RISK\s+FACTORS',
            r'ITEM\s*1A[:\-\s]+RISK'
        ],
        'Item 1 - Business': [
            r'ITEM\s*1[\.\s]+BUSINESS',
            r'ITEM\s*1[:\-\s]+BUSINESS',
            r'^ITEM\s*1[\.\s](?!A|B|C)'
        ],
        'Item 1B': [r'ITEM\s*1B'],
        'Item 1C - Cybersecurity': [
            r'ITEM\s*1C',
            r'CYBERSECURITY'
        ],
        'Item 2 - Properties': [r'ITEM\s*2[\.\s:]'],
        'Item 3 - Legal Proceedings': [
            r'ITEM\s*3[\.\s:]',
            r'LEGAL\s+PROCEEDINGS'
        ],
        'Item 7 - MD&A': [
            r'ITEM\s*7[\.\s](?!A)',
            r"MANAGEMENT'?S\s+DISCUSSION",
            r'MD&A'
        ],
        'Item 7A - Market Risk': [
            r'ITEM\s*7A',
            r'MARKET\s+RISK'
        ],
        'Item 8 - Financial Statements': [
            r'ITEM\s*8[\.\s:]',
            r'FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY',
            r'CONSOLIDATED\s+BALANCE\s+SHEET',
            r'CONSOLIDATED\s+STATEMENT.*OPERATIONS',
            r'CONSOLIDATED\s+STATEMENT.*CASH\s+FLOW'
        ],
        'Item 9A - Controls and Procedures': [r'ITEM\s*9A'],
        'Item 10 - Directors and Officers': [
            r'ITEM\s*10[\.\s:]',
            r'DIRECTORS.*EXECUTIVE\s+OFFICERS',
            r'EXECUTIVE\s+OFFICERS.*DIRECTORS',
            r'SIGNATURES',
            r'/s/',
        ],
        'Item 11 - Executive Compensation': [
            r'ITEM\s*11',
            r'EXECUTIVE\s+COMPENSATION'
        ],
        'Item 15 - Exhibits': [r'ITEM\s*15']
    }
    
    # Check each pattern
    for section, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, text_upper):
                return section
    
    # Check for financial statement notes
    note_match = re.search(r'NOTE\s+(\d+)', text_upper)
    if note_match:
        return f'Note {note_match.group(1)}'
    
    # Check for Parts
    if 'PART I' in text_upper and 'PART II' not in text_upper:
        return 'Part I'
    elif 'PART II' in text_upper and 'PART III' not in text_upper:
        return 'Part II'
    elif 'PART III' in text_upper and 'PART IV' not in text_upper:
        return 'Part III'
    elif 'PART IV' in text_upper:
        return 'Part IV'
    
    return 'Unknown'

def detect_subsection(text: str) -> str:
    """Detect subsection within a section."""
    text_upper = text.upper()
    
    # Business subsections
    if 'BUSINESS OVERVIEW' in text_upper or 'ABOUT US' in text_upper:
        return 'Business Overview'
    elif 'COMPETITION' in text_upper and 'RISK' not in text_upper:
        return 'Competition'
    elif 'REGULATION' in text_upper or 'REGULATORY' in text_upper:
        return 'Regulation'
    elif 'HUMAN CAPITAL' in text_upper or 'EMPLOYEES' in text_upper:
        return 'Human Capital'
    elif 'INTELLECTUAL PROPERTY' in text_upper:
        return 'Intellectual Property'
    
    # MD&A subsections
    if 'LIQUIDITY' in text_upper and 'CAPITAL' in text_upper:
        return 'Liquidity and Capital Resources'
    elif 'RESULTS OF OPERATIONS' in text_upper:
        return 'Results of Operations'
    elif 'CRITICAL ACCOUNTING' in text_upper:
        return 'Critical Accounting Estimates'
    
    # Financial statement subsections
    if 'CONSOLIDATED STATEMENTS OF OPERATIONS' in text_upper:
        return 'Income Statement'
    elif 'CONSOLIDATED BALANCE SHEET' in text_upper:
        return 'Balance Sheet'
    elif 'CONSOLIDATED STATEMENTS OF CASH FLOWS' in text_upper:
        return 'Cash Flow Statement'
    elif 'STOCKHOLDERS EQUITY' in text_upper or "SHAREHOLDERS' EQUITY" in text_upper:
        return 'Equity Statement'
    
    return ''

def detect_table(text: str) -> bool:
    """Detect if chunk contains tabular data."""
    patterns = [
        r'\$\s*[\d,]+',
        r'\d{1,3}(,\d{3})+',
        r'(\s{2,}|\t)\d+(\.\d+)?(\s{2,}|\t)',
        r'^\s*\d{4}\s+\d{4}\s+\d{4}',
        r'─{3,}|═{3,}|_{3,}',
    ]
    
    match_count = sum(1 for pattern in patterns if re.search(pattern, text, re.MULTILINE))
    return match_count >= 2

def detect_numbers(text: str) -> bool:
    """Detect if chunk contains significant numerical data."""
    dollar_matches = len(re.findall(r'\$\s*[\d,]+', text))
    percent_matches = len(re.findall(r'\d+\.?\d*\s*%', text))
    return (dollar_matches + percent_matches) >= 3

def detect_forward_looking(text: str) -> bool:
    """Detect forward-looking statements."""
    forward_keywords = [
        'expect', 'anticipate', 'believe', 'estimate', 'intend',
        'may', 'will', 'should', 'could', 'would', 'plan',
        'forecast', 'projection', 'outlook', 'guidance', 'target'
    ]
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in forward_keywords if keyword in text_lower)
    return keyword_count >= 2

def detect_legal_matter(text: str) -> bool:
    """Detect legal proceedings and litigation content."""
    legal_keywords = [
        'litigation', 'lawsuit', 'plaintiff', 'defendant', 'settlement',
        'class action', 'legal proceedings', 'arbitration', 'judgment',
        'damages', 'indemnification', 'contingent liabilities'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in legal_keywords)

def detect_director_info(text: str) -> bool:
    """Detect content about directors and officers."""
    director_keywords = [
        'director', 'officer', 'board of directors', 'executive officer',
        'chief executive', 'chief financial', 'president', 'vice president',
        'compensation committee', 'audit committee', 'governance', '/s/'
    ]
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in director_keywords if keyword in text_lower)
    
    # Also check for signature pattern
    has_signature = bool(re.search(r'/s/\s*[A-Za-z]', text))
    
    return keyword_count >= 2 or has_signature

def detect_risk_category(text: str) -> str:
    """Detect the category of risk factor."""
    text_lower = text.lower()
    
    if 'cybersecurity' in text_lower or 'data breach' in text_lower or 'cyber attack' in text_lower:
        return 'Cybersecurity'
    elif 'regulation' in text_lower or 'regulatory' in text_lower or 'compliance' in text_lower:
        return 'Regulatory'
    elif 'competition' in text_lower or 'competitive' in text_lower:
        return 'Competition'
    elif 'technology' in text_lower or 'technical' in text_lower or 'system' in text_lower:
        return 'Technology'
    elif 'litigation' in text_lower or 'legal' in text_lower:
        return 'Litigation'
    elif 'financial' in text_lower or 'liquidity' in text_lower or 'debt' in text_lower:
        return 'Financial'
    elif 'operational' in text_lower or 'operations' in text_lower:
        return 'Operational'
    elif 'international' in text_lower or 'foreign' in text_lower:
        return 'International'
    elif 'intellectual property' in text_lower or 'patent' in text_lower or 'trademark' in text_lower:
        return 'Intellectual Property'
    elif 'privacy' in text_lower or 'data protection' in text_lower:
        return 'Privacy'
    
    return ''

def detect_note_topic(text: str) -> str:
    """Detect the topic of a financial statement note."""
    text_lower = text.lower()
    
    if 'accounting policies' in text_lower or 'significant accounting' in text_lower:
        return 'Accounting Policies'
    elif 'revenue' in text_lower and 'recognition' in text_lower:
        return 'Revenue Recognition'
    elif 'debt' in text_lower or 'borrowing' in text_lower or 'credit facility' in text_lower:
        return 'Debt'
    elif 'derivative' in text_lower or 'hedging' in text_lower:
        return 'Derivatives and Hedging'
    elif 'income tax' in text_lower or 'tax provision' in text_lower:
        return 'Income Taxes'
    elif 'leases' in text_lower or 'lease obligations' in text_lower:
        return 'Leases'
    elif 'stockholders equity' in text_lower or 'share repurchase' in text_lower or 'stock-based compensation' in text_lower:
        return 'Stockholders Equity'
    elif 'commitments and contingencies' in text_lower:
        return 'Commitments and Contingencies'
    elif 'segment' in text_lower or 'geographic' in text_lower:
        return 'Segment Information'
    elif 'acquisitions' in text_lower or 'business combination' in text_lower:
        return 'Acquisitions'
    elif 'content' in text_lower and ('asset' in text_lower or 'amortization' in text_lower):
        return 'Content Assets'
    elif 'legal' in text_lower and 'matter' in text_lower:
        return 'Legal Matters'
    elif 'fair value' in text_lower:
        return 'Fair Value'
    elif 'pension' in text_lower or 'benefit plan' in text_lower or '401(k)' in text_lower:
        return 'Employee Benefits'
    
    return ''

def detect_geographic_region(text: str) -> str:
    """Detect geographic region mentioned in text."""
    text_upper = text.upper()
    
    if 'NORTH AMERICA' in text_upper or 'UNITED STATES AND CANADA' in text_upper or 'UCAN' in text_upper:
        return 'North America'
    elif 'EMEA' in text_upper or 'EUROPE, MIDDLE EAST' in text_upper:
        return 'EMEA'
    elif 'LATIN AMERICA' in text_upper or 'LATAM' in text_upper:
        return 'Latin America'
    elif 'ASIA' in text_upper or 'ASIA-PACIFIC' in text_upper or 'APAC' in text_upper:
        return 'Asia-Pacific'
    elif 'INTERNATIONAL' in text_upper:
        return 'International'
    
    return ''

def load_10k_filings(
    filings_dir: str = FILINGS_DIR,
    pattern: str = "**/*.txt"
) -> List[Document]:
    """
    Load all 10-K filings from directory structure.
    
    Args:
        filings_dir: Root directory containing ticker subdirectories
        pattern: Glob pattern for finding filing files
    
    Returns:
        List of LangChain Documents with metadata
    
    Raises:
        FileNotFoundError: If directory doesn't exist or no files found
        ValueError: If directory is empty
    """
    filings_path = Path(filings_dir)
    
    # Check if directory exists
    if not filings_path.exists():
        raise FileNotFoundError(f"Directory not found: {filings_dir}")
    
    # Check if directory is empty
    if not any(filings_path.iterdir()):
        raise ValueError(f"Directory is empty: {filings_dir}")
    
    # Find all .txt files
    filing_files = list(filings_path.glob(pattern))
    
    if not filing_files:
        raise FileNotFoundError(
            f"No .txt files found in {filings_dir}. "
            f"Expected structure: {filings_dir}/TICKER/TICKER_10-K_DATE_ACCESSION.txt"
        )
    
    logger.info(f"Found {len(filing_files)} filing files")
    
    documents = []
    failed_files = []
    
    for file_path in tqdm(filing_files, desc="Loading filings"):
        try:
            # Extract metadata from path
            metadata = extract_metadata_from_path(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Validate content
            if not content or len(content.strip()) < 100:
                logger.warning(f"File appears empty or too short: {file_path}")
                failed_files.append((file_path, "Empty or too short"))
                continue
            
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {file_path}: {e}")
            failed_files.append((file_path, f"Encoding error: {e}"))
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            failed_files.append((file_path, str(e)))
    
    # Report results
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files:")
        for file_path, error in failed_files:
            logger.warning(f"  {file_path.name}: {error}")
    
    if not documents:
        raise ValueError(
            f"No documents successfully loaded from {filings_dir}. "
            f"Check file format and encoding."
        )
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, 
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Chunk documents using RecursiveCharacterTextSplitter with custom separators
    optimized for SEC 10-K filings.
    """
    # Custom separators prioritizing SEC document structure
    separators = [
        "\n\nNote ",          # Financial notes boundaries
        "\n\nNOTE ",          # Alternate capitalization
        "\n\nITEM ",          # SEC-mandated section breaks
        "\n\nItem ",          # Alternate capitalization
        "\n\nPART ",          # PART I, II, III, IV divisions
        "\n\nPart ",          # Alternate capitalization
        "\n\n  ",            # Indented subsections
        "\n\n",              # Paragraph breaks
        "\n",                # Line breaks
        ". ",                # Sentences
        " ",                 # Words
        ""                   # Characters
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False
    )
    
    chunked_docs = []
    
    MIN_CHUNK_SIZE = 100  # Minimum 100 characters
    
    for doc in tqdm(documents, desc="Chunking documents"):
        # Detect if this is an Officers and Directors section
        is_officer_section = 'ITEM 10' in doc.page_content.upper() or \
                           'DIRECTORS' in doc.page_content.upper() or \
                           'EXECUTIVE OFFICERS' in doc.page_content.upper()
        
        if is_officer_section:
            # Use smaller chunk size for officer/director sections
            officer_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=separators,
                is_separator_regex=False
            )
            chunks = officer_splitter.split_documents([doc])
        else:
            chunks = text_splitter.split_documents([doc])
        
        # Enhance metadata for each chunk
        for i, chunk in enumerate(chunks):
            
            if len(chunk.page_content.strip()) < MIN_CHUNK_SIZE:
                continue
            
            else:
                chunk.metadata['chunk_id'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                
                # Add section detection
                chunk.metadata['section'] = detect_section(chunk.page_content)
                chunk.metadata['subsection'] = detect_subsection(chunk.page_content)
                
                # Add content type flags
                chunk.metadata['contains_table'] = detect_table(chunk.page_content)
                chunk.metadata['contains_numbers'] = detect_numbers(chunk.page_content)
                chunk.metadata['is_forward_looking'] = detect_forward_looking(chunk.page_content)
                
                # Add company-specific flags
                chunk.metadata['is_legal_matter'] = detect_legal_matter(chunk.page_content)
                chunk.metadata['contains_director_info'] = detect_director_info(chunk.page_content)
                
                # Extract note number if in Item 8
                if 'Item 8' in chunk.metadata['section'] or 'Financial' in chunk.metadata['section']:
                    note_match = re.search(r'Note\s+(\d+)', chunk.page_content, re.IGNORECASE)
                    if note_match:
                        chunk.metadata['note_number'] = f"Note {note_match.group(1)}"
                        chunk.metadata['note_topic'] = detect_note_topic(chunk.page_content)
                
                # Add risk category for Item 1A
                if 'Item 1A' in chunk.metadata['section']:
                    risk_cat = detect_risk_category(chunk.page_content)
                    if risk_cat:
                        chunk.metadata['risk_category'] = risk_cat
                
                # Add geographic region if detected
                region = detect_geographic_region(chunk.page_content)
                if region:
                    chunk.metadata['geographic_region'] = region
                
                chunked_docs.append(chunk)
    
    return chunked_docs

def create_vector_store(documents: List[Document], persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """
    Create and persist a Chroma vector store from documents.
    Uses Google Generative AI embeddings with batch processing.
    """
    logger.info("Creating embeddings with Google Generative AI...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Process in batches to avoid timeout
    BATCH_SIZE = 100  # Process 100 chunks at a time
    total_docs = len(documents)
    
    logger.info(f"Creating vector store with {total_docs} chunks...")
    logger.info(f"Processing in batches of {BATCH_SIZE}...")
    
    vectorstore = None
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        try:
            if vectorstore is None:
                # Create the vector store with the first batch
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=persist_dir,
                    collection_name="sec_filings"
                )
            else:
                # Add subsequent batches to existing store
                vectorstore.add_documents(batch)
            
            logger.info(f"  ✓ Batch {batch_num} completed")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing batch {batch_num}: {e}")
            # Continue with next batch instead of failing completely
            continue
    
    if vectorstore is None:
        raise RuntimeError("Failed to create vector store - all batches failed")
    
    logger.info(f"Vector store created and persisted to {persist_dir}")
    logger.info(f"Final count: {vectorstore._collection.count()} chunks")
    
    return vectorstore

import time
from typing import Optional

def create_vector_store_with_retry(
    documents: List[Document], 
    persist_dir: str = CHROMA_PERSIST_DIR,
    batch_size: int = 100,
    max_retries: int = 3
) -> Chroma:
    """
    Create vector store with automatic retry on failures.
    """
    logger.info("Creating embeddings with Google Generative AI...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    logger.info(f"Creating vector store with {total_docs} chunks...")
    logger.info(f"Processing in {total_batches} batches of {batch_size}...")
    
    vectorstore: Optional[Chroma] = None
    failed_batches = []
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        # Try with retries
        success = False
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    # Create the vector store with the first batch
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=persist_dir,
                        collection_name="sec_filings"
                    )
                else:
                    # Add subsequent batches
                    vectorstore.add_documents(batch)
                
                logger.info(f"  ✓ Batch {batch_num} completed")
                success = True
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    logger.warning(f"  ⚠ Attempt {attempt + 1} failed: {e}")
                    logger.warning(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  ✗ Batch {batch_num} failed after {max_retries} attempts: {e}")
                    failed_batches.append(batch_num)
        
        # Small delay between batches to avoid rate limits
        if success and i + batch_size < total_docs:
            time.sleep(1)
    
    if vectorstore is None:
        raise RuntimeError("Failed to create vector store - first batch failed")
    
    # Report results
    final_count = vectorstore._collection.count()
    logger.info(f"\nVector store creation complete:")
    logger.info(f"  ✓ Successfully processed: {final_count} chunks")
    logger.info(f"  ✓ Persisted to: {persist_dir}")
    
    if failed_batches:
        logger.warning(f"  ⚠ Failed batches: {len(failed_batches)} ({failed_batches})")
        logger.warning(f"  Expected: {total_docs}, Got: {final_count}")
    
    return vectorstore

def main():
    """
    Main execution function.
    """
    logger.info("=" * 80)
    logger.info("Starting 10-K processing pipeline...")
    logger.info("=" * 80)
    
    try:
        # Load documents
        logger.info(f"\n1. Loading documents from {FILINGS_DIR}...")
        documents = load_10k_filings(FILINGS_DIR)
        logger.info(f"   ✓ Loaded {len(documents)} document(s)")
        
        # Log company summary
        companies = set(doc.metadata.get('ticker', 'Unknown') for doc in documents)
        logger.info(f"   ✓ Companies: {', '.join(sorted(companies))}")
        
        # Chunk documents
        logger.info(f"\n2. Chunking documents...")
        logger.info(f"   - Chunk size: {CHUNK_SIZE} characters")
        logger.info(f"   - Chunk overlap: {CHUNK_OVERLAP} characters")
        chunks = chunk_documents(documents)
        logger.info(f"   ✓ Created {len(chunks)} chunks")
        
        # Log sample chunk metadata
        if chunks:
            sample = chunks[0]
            logger.info(f"\n3. Sample chunk metadata:")
            for key, value in sample.metadata.items():
                logger.info(f"     {key}: {value}")
        
        # Create vector store with retry logic
        logger.info(f"\n4. Creating vector store...")
        vectorstore = create_vector_store_with_retry(
            chunks,
            batch_size=100,  # Adjust based on API limits
            max_retries=3
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Processing complete!")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ Error: {e}")
        logger.error("Please check that:")
        logger.error(f"  1. Directory exists: {FILINGS_DIR}")
        logger.error(f"  2. Directory structure: {FILINGS_DIR}/TICKER/TICKER_10-K_DATE_ACCESSION.txt")
        logger.error(f"  3. Files have .txt extension")
        return 1
        
    except ValueError as e:
        logger.error(f"\n✗ Error: {e}")
        return 1
        
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())