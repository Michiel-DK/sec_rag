#!/usr/bin/env python3
"""
Bulk 10-K Filing Downloader using edgartools

This script downloads 10-K filings for a list of companies from the SEC EDGAR database.
"""

from edgar import *
import os
import time
from pathlib import Path

# Set your identity for SEC EDGAR (REQUIRED by SEC)
# Replace with your actual information
set_identity(os.getenv('EDGAR_IDENTITY'))


def download_10k_filings(
    companies,
    output_dir="10k_filings",
    filing_type="10-K",
    num_filings=5,
    delay=0.1
):
    """
    Download 10-K filings for multiple companies.
    
    Args:
        companies: List of tickers or (ticker, cik) tuples
        output_dir: Directory to save filings
        filing_type: Type of filing (default: "10-K")
        num_filings: Number of most recent filings to download per company
        delay: Delay between requests in seconds (SEC rate limit: 10 req/sec)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        "success": [],
        "failed": [],
        "no_filings": []
    }
    
    for i, company_info in enumerate(companies, 1):
        # Handle different input formats
        if isinstance(company_info, tuple):
            ticker = company_info[0]
        else:
            ticker = company_info
        
        print(f"\n[{i}/{len(companies)}] Processing {ticker}...")
        
        try:
            # Create company directory
            company_dir = output_path / ticker
            company_dir.mkdir(exist_ok=True)
            
            # Get company object
            company = Company(ticker)
            print(f"  Company: {company.name}")
            
            # Get filings
            print(f"  Fetching {filing_type} filings...")
            filings_result = company.get_filings(form=filing_type).latest(num_filings)
            
            # Handle single filing vs multiple filings
            if num_filings == 1:
                # latest(1) returns a single EntityFiling object
                if not filings_result:
                    print(f"  ⚠️  No {filing_type} filings found for {ticker}")
                    results["no_filings"].append(ticker)
                    continue
                filings = [filings_result]  # Wrap in list
            else:
                # latest(n) where n>1 returns a Filings collection
                filings = list(filings_result)
                if not filings or len(filings) == 0:
                    print(f"  ⚠️  No {filing_type} filings found for {ticker}")
                    results["no_filings"].append(ticker)
                    continue
            
            print(f"  Found filings, downloading {len(filings)}...")
            
            # Download each filing
            for j, filing in enumerate(filings, 1):
                filing_date = filing.filing_date
                accession = filing.accession_no.replace("-", "")
                filename = f"{ticker}_{filing_type}_{filing_date}_{accession[:10]}.txt"
                filepath = company_dir / filename
                
                # Get the filing text
                text = filing.text()
                
                # Save to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"    ✓ Downloaded: {filename}")
                
                # Respect SEC rate limits
                time.sleep(delay)
            
            results["success"].append(ticker)
            print(f"  ✓ Completed {ticker}: {len(filings)} filings downloaded")
            
        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {str(e)}")
            results["failed"].append((ticker, str(e)))
        
        # Small delay between companies
        time.sleep(delay)
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✓ Successfully downloaded: {len(results['success'])} companies")
    print(f"⚠️  No filings found: {len(results['no_filings'])} companies")
    print(f"✗ Failed: {len(results['failed'])} companies")
    
    if results["failed"]:
        print("\nFailed companies:")
        for ticker, error in results["failed"]:
            print(f"  - {ticker}: {error}")
    
    if results["no_filings"]:
        print("\nCompanies with no filings:")
        for ticker in results["no_filings"]:
            print(f"  - {ticker}")
    
    return results


def main():
    """Example usage"""
    
    # List of ticker symbols
    companies = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet
        "AMZN",  # Amazon
        "TSLA",  # Tesla
    ]
    
    print("Bulk 10-K Filing Downloader")
    print("="*60)
    print("\nIMPORTANT: Update set_identity() with your information")
    print("before running this script. The SEC requires user identification.")
    print("\n" + "="*60)
    
    # Download filings
    results = download_10k_filings(
        companies=companies,
        output_dir="10k_filings",
        filing_type="10-K",
        num_filings=3,  # Download 3 most recent 10-Ks per company
        delay=0.15  # 150ms delay between requests
    )
    
    print(f"\nAll filings saved to: {Path('10k_filings').absolute()}")


if __name__ == "__main__":
    main()
