# Bulk 10-K Filing Downloader

A Python toolkit for bulk downloading 10-K filings from the SEC EDGAR database using the `edgartools` library.

## Installation

```bash
pip install edgartools
pip install pandas  # For CSV functionality
```

## ⚠️ IMPORTANT: SEC Requirements

The SEC **requires** you to identify yourself when accessing EDGAR. Before running any scripts, you must update this line with your actual information:

```python
set_identity("Your Name your.email@example.com")
```

**Example:**
```python
set_identity("John Doe john.doe@company.com")
```

Failure to provide proper identification may result in being blocked by the SEC.

## Quick Start

### Option 1: Download from a Python List

```python
from edgar import *

# Set your identity (REQUIRED)
set_identity("Your Name your.email@example.com")

# Define companies
companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Download
from bulk_10k_downloader import download_10k_filings
results = download_10k_filings(
    companies=companies,
    output_dir="10k_filings",
    num_filings=3  # Download 3 most recent 10-Ks per company
)
```

### Option 2: Download from a CSV File

Create a CSV file (`companies.csv`) with your company list:

```csv
ticker,cik,company_name
AAPL,0000320193,Apple Inc.
MSFT,0000789019,Microsoft Corporation
GOOGL,0001652044,Alphabet Inc.
```

Then run:

```python
from bulk_10k_from_csv import download_10k_from_csv

results = download_10k_from_csv(
    csv_file="companies.csv",
    output_dir="10k_filings",
    num_filings=3
)
```

## Features

- ✅ Bulk download 10-K filings for multiple companies
- ✅ Support for ticker symbols or CIK numbers
- ✅ CSV file input support
- ✅ Configurable number of filings per company
- ✅ Automatic rate limiting (respects SEC 10 req/sec limit)
- ✅ Error handling and progress tracking
- ✅ Download summary reports
- ✅ Organized output by company

## Usage Examples

### Example 1: Basic Usage with Tickers

```python
companies = ["AAPL", "MSFT", "GOOGL"]

results = download_10k_filings(
    companies=companies,
    output_dir="filings",
    num_filings=5
)
```

### Example 2: Using CIK Numbers for Precision

```python
# Format: (ticker, cik)
companies = [
    ("AAPL", "0000320193"),
    ("MSFT", "0000789019"),
    ("GOOGL", "0001652044"),
]

results = download_10k_filings(companies=companies)
```

### Example 3: Download Different Filing Types

```python
# Download 10-Q filings instead of 10-K
results = download_10k_filings(
    companies=["AAPL", "MSFT"],
    filing_type="10-Q",
    num_filings=4
)

# Download 8-K filings
results = download_10k_filings(
    companies=["AAPL", "MSFT"],
    filing_type="8-K",
    num_filings=10
)
```

### Example 4: Adjust Rate Limiting

```python
# Faster (but still within SEC limits)
results = download_10k_filings(
    companies=companies,
    delay=0.11  # ~9 requests per second
)

# More conservative (safer for large batches)
results = download_10k_filings(
    companies=companies,
    delay=0.5  # 2 requests per second
)
```

## Output Structure

```
10k_filings/
├── AAPL/
│   ├── AAPL_10-K_2023-11-03.txt
│   ├── AAPL_10-K_2022-10-28.txt
│   └── AAPL_10-K_2021-10-29.txt
├── MSFT/
│   ├── MSFT_10-K_2023-08-01.txt
│   ├── MSFT_10-K_2022-07-28.txt
│   └── MSFT_10-K_2021-07-29.txt
└── download_summary.csv (when using CSV input)
```

## Finding CIK Numbers

You can find company CIK numbers at:
- https://www.sec.gov/cgi-bin/browse-edgar
- Or use the ticker symbol and let the library look it up

## Available Filing Types

Common filing types you can download:
- `10-K` - Annual report
- `10-Q` - Quarterly report
- `8-K` - Current report (material events)
- `DEF 14A` - Proxy statement
- `S-1` - Registration statement
- `20-F` - Annual report (foreign issuers)

## Rate Limiting

The SEC limits requests to **10 per second**. The scripts include automatic rate limiting:
- Default delay: 0.15 seconds (≈6.7 requests/second)
- Adjust the `delay` parameter as needed
- Going too fast may result in your IP being blocked

## Error Handling

The scripts handle common errors:
- Companies with no filings
- Network errors
- Invalid ticker symbols
- Rate limit issues

Results are categorized as:
- `success`: Downloaded successfully
- `no_filings`: Company exists but no filings found
- `failed`: Error occurred during download

## Tips for Large Downloads

1. **Start small**: Test with 5-10 companies first
2. **Use CIK numbers**: More reliable than tickers
3. **Increase delay**: Use `delay=0.5` for very large batches (thousands of companies)
4. **Monitor progress**: Scripts print real-time progress
5. **Check summaries**: Review the summary output for any failures

## Troubleshooting

### "No filings found"
- Verify the ticker symbol is correct
- Check if the company files the requested form type
- Try using the CIK number instead

### "Rate limit exceeded" or IP blocked
- Increase the `delay` parameter
- Wait 30 minutes before retrying
- Ensure you've set your identity properly

### "Company not found"
- Double-check the ticker symbol
- Try looking up the CIK number manually
- The company may have changed tickers or been delisted

## Advanced: Parallel Downloads

For very large lists (thousands of companies), consider:

```python
from concurrent.futures import ThreadPoolExecutor
import time

def download_single_company(company):
    time.sleep(0.15)  # Rate limiting
    # Your download code here
    pass

# Use with caution - ensure total rate stays under 10 req/sec
with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(download_single_company, companies)
```

## References

- [edgartools PyPI](https://pypi.org/project/edgartools/)
- [edgartools Documentation](https://edgartools.readthedocs.io/)
- [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch.html)
- [SEC Fair Access Policy](https://www.sec.gov/os/webmaster-faq#code-support)