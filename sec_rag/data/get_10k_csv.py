#!/usr/bin/env python3
"""
Bulk 10-K Downloader from CSV

This script reads a CSV file with company information and downloads 10-K filings.
CSV should have columns: ticker, cik (optional), company_name (optional)
"""

from edgar import *
import pandas as pd
import os
import time
from pathlib import Path

# Set your identity for SEC EDGAR (REQUIRED)
set_identity("Your Name your.email@example.com")


def download_10k_from_csv(
    csv_file,
    output_dir="10k_filings",
    filing_type="10-K",
    num_filings=5,
    delay=0.15
):
    """
    Download 10-K filings for companies listed in a CSV file.
    
    Args:
        csv_file: Path to CSV file with company information
        output_dir: Directory to save filings
        filing_type: Type of filing (default: "10-K")
        num_filings: Number of recent filings per company
        delay: Delay between requests in seconds
    """
    # Read CSV file
    print(f"Reading companies from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Validate required columns
    if 'ticker' not in df.columns:
        raise ValueError("CSV must have a 'ticker' column")
    
    print(f"Found {len(df)} companies in CSV")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare company list  
    companies = []
    for _, row in df.iterrows():
        ticker = row['ticker']
        companies.append(ticker)
    
    # Download filings
    results = {
        "success": [],
        "failed": [],
        "no_filings": []
    }
    
    for i, ticker in enumerate(companies, 1):
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
                time.sleep(delay)
            
            results["success"].append(ticker)
            print(f"  ✓ Completed {ticker}")
            
        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {str(e)}")
            results["failed"].append((ticker, str(e)))
        
        time.sleep(delay)
    
    # Save summary report
    summary_df = pd.DataFrame({
        'ticker': [t for t in results['success']] + 
                  [t for t in results['no_filings']] + 
                  [t for t, _ in results['failed']],
        'status': (['success'] * len(results['success']) + 
                   ['no_filings'] * len(results['no_filings']) + 
                   ['failed'] * len(results['failed'])),
        'error': [None] * len(results['success']) + 
                 [None] * len(results['no_filings']) + 
                 [err for _, err in results['failed']]
    })
    
    summary_file = output_path / 'download_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✓ Successfully downloaded: {len(results['success'])} companies")
    print(f"⚠️  No filings found: {len(results['no_filings'])} companies")
    print(f"✗ Failed: {len(results['failed'])} companies")
    print(f"\nSummary saved to: {summary_file}")
    
    return results


def create_sample_csv():
    """Create a sample CSV file for demonstration"""
    sample_data = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'cik': ['0000320193', '0000789019', '0001652044', '0001018724', '0001318605'],
        'company_name': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 
                         'Amazon.com Inc.', 'Tesla Inc.']
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('companies.csv', index=False)
    print("Created sample CSV file: companies.csv")
    return 'companies.csv'


def main():
    print("Bulk 10-K Downloader from CSV")
    print("="*60)
    print("\nIMPORTANT: Update set_identity() with your information")
    print("before running this script.\n")
    
    # Create sample CSV if it doesn't exist
    csv_file = 'companies.csv'
    if not os.path.exists(csv_file):
        print("No companies.csv found. Creating sample file...")
        csv_file = create_sample_csv()
    
    # Download filings
    results = download_10k_from_csv(
        csv_file=csv_file,
        output_dir="10k_filings",
        filing_type="10-K",
        num_filings=3,
        delay=0.15
    )
    
    print(f"\nAll filings saved to: {Path('10k_filings').absolute()}")


if __name__ == "__main__":
    main()
