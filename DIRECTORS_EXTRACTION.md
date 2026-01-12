# Director and Officer Extraction

## Overview

The SEC RAG system now includes functionality to extract structured information about directors, officers, and executives from 10-K filings. This feature automatically identifies and categorizes key personnel, making it easy to understand company leadership.

## Features

### Extraction Capabilities

The system can extract:
- **Names**: Full names of directors and officers
- **Titles**: Complete job titles and roles
- **Dates**: Filing/signature dates
- **Classifications**:
  - Executive Officers (CEO, CFO, etc.)
  - Board of Directors
  - Other Officers

### Supported Formats

The extraction handles multiple 10-K formats:
1. **Signature Blocks**: Standard signature pages with `/s/` markers
2. **Tables**: Structured tables with Name, Title, Date columns
3. **Lists**: Comma-separated or bulleted lists

### Example Input

```
  /s/ Timothy D. Cook         Chief Executive Officer and Director                October 31, 2025
  /s/ Kevan Parekh            Senior Vice President, Chief Financial Officer      October 31, 2025
  /s/ Arthur D. Levinson      Director and Chair of the Board                     October 31, 2025
```

### Example Output

```
**AAPL - Directors and Officers**

**Executive Officers:**
• Timothy D. Cook - Chief Executive Officer and Director (Principal Executive Officer)
• Kevan Parekh - Senior Vice President, Chief Financial Officer

**Directors:**
• Arthur D. Levinson - Director and Chair of the Board
• Wanda Austin - Director
• Alex Gorsky - Director

**Total:** 10 people
```

## Usage

### Agent Tool

The simplest way to extract director information is through the agent:

```python
from sec_rag.agent.similarity_agent import create_agent

agent = create_agent()
response = agent.run("Who are the directors and officers of Apple?")
print(response)
```

**Supported queries:**
- "Who are the directors and officers of AAPL?"
- "Show me the executive team at Microsoft"
- "Get directors for Tesla"
- "List NVDA's board of directors"

### Direct API

You can also use the extraction functions directly:

```python
from sec_rag.chroma.extract_people import extract_people, format_people_output

# Extract from text
text = """
  /s/ Timothy D. Cook    Chief Executive Officer    October 31, 2025
"""

people = extract_people(text)

# Format for display
output = format_people_output(people, ticker="AAPL")
print(output)
```

### Programmatic Access

For structured data access:

```python
from sec_rag.chroma.extract_people import extract_people

people = extract_people(text)

for person in people:
    print(f"Name: {person.name}")
    print(f"Title: {person.title}")
    print(f"Is Executive: {person.is_executive}")
    print(f"Is Director: {person.is_director}")
    print(f"Is Officer: {person.is_officer}")
    print(f"Date: {person.date}")
    print()
```

## Technical Details

### Person Dataclass

Each extracted person is represented as:

```python
@dataclass
class Person:
    name: str                  # Full name (title cased)
    title: str                 # Complete job title
    date: Optional[str]        # Filing/signature date
    signature: bool            # Has /s/ signature marker
    is_director: bool          # Board member
    is_officer: bool           # Officer (CEO, CFO, etc.)
    is_executive: bool         # Executive officer
```

### Classification Logic

**Director**: Title contains "director"
**Officer**: Title contains officer keywords (CEO, CFO, President, VP, etc.)
**Executive**: Title contains executive keywords (Chief, CEO, CFO, President, etc.)

### Chunking Strategy

The system uses specialized chunking for director/officer sections:
- **Smaller chunks** (1800 chars) for Item 10 sections
- **Signature detection** in metadata (`contains_director_info`)
- **Pattern matching** for sections with `/s/` markers

This ensures that director information is preserved across chunk boundaries.

## Files

- `sec_rag/chroma/extract_people.py` - Core extraction logic
- `sec_rag/agent/tools.py` - Agent tool definition
- `sec_rag/chroma/load_filings_to_chroma.py` - Enhanced chunking

## Testing

Test the extraction:

```bash
# Test extraction logic
python -m sec_rag.chroma.test_extract_people

# Test agent tool
python -m sec_rag.agent.test_directors_tool

# Test full agent queries
python -m sec_rag.agent.test_directors_query
```

## Limitations

1. **Format dependency**: Works best with standard 10-K signature blocks
2. **Name variations**: May have issues with unusual name formats
3. **Classification accuracy**: Title-based classification may not catch all roles
4. **Chunking sensitive**: Quality depends on proper chunking of Item 10 sections

## Future Improvements

- [ ] Support for biographical information extraction
- [ ] Committee membership extraction
- [ ] Historical comparison (track changes over time)
- [ ] Cross-reference with other SEC filings (proxy statements)
- [ ] Entity resolution (same person across companies)
- [ ] Compensation data extraction
