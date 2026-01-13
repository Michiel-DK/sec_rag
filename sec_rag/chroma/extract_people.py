"""
Extract structured information about directors, officers, and executives from SEC 10-K filings.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Person:
    """Represents a director or officer from a 10-K filing."""
    name: str
    title: str
    date: Optional[str] = None
    signature: bool = False
    is_director: bool = False
    is_officer: bool = False
    is_executive: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'title': self.title,
            'date': self.date,
            'signature': self.signature,
            'is_director': self.is_director,
            'is_officer': self.is_officer,
            'is_executive': self.is_executive
        }


def clean_name(name: str) -> str:
    """Clean and standardize a person's name."""
    # Remove signature markers
    name = re.sub(r'/s/', '', name, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Remove leading/trailing punctuation
    name = name.strip('.,;:')
    
    # Title case
    name = name.title()
    
    return name.strip()


def clean_title(title: str) -> str:
    """Clean and standardize a title."""
    # Remove extra whitespace and parentheses content on separate lines
    lines = title.split('\n')
    title = ' '.join(line.strip() for line in lines if line.strip())
    title = ' '.join(title.split())
    
    return title.strip()


def classify_role(title: str) -> Dict[str, bool]:
    """Classify a person's role based on their title."""
    title_lower = title.lower()
    
    classification = {
        'is_director': False,
        'is_officer': False,
        'is_executive': False
    }
    
    # Check for director
    if 'director' in title_lower:
        classification['is_director'] = True
    
    # Check for officer
    officer_keywords = ['officer', 'ceo', 'cfo', 'coo', 'cto', 'president', 'vice president']
    if any(keyword in title_lower for keyword in officer_keywords):
        classification['is_officer'] = True
    
    # Check for executive
    executive_keywords = ['chief', 'ceo', 'cfo', 'coo', 'cto', 'executive', 'president']
    if any(keyword in title_lower for keyword in executive_keywords):
        classification['is_executive'] = True
    
    return classification


def extract_people_from_signature_block(text: str) -> List[Person]:
    """
    Extract people from signature block format (most common in 10-K).
    
    Example format:
        /s/ Timothy D. Cook         Chief Executive Officer and Director                October 31, 2025
        TIMOTHY D. COOK
    """
    people = []
    
    # Split into lines
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for signature lines (starting with /s/)
        if '/s/' in line.lower():
            # Extract components using flexible regex
            # Pattern: /s/ NAME   TITLE   DATE
            match = re.search(
                r'/s/\s*([A-Za-z\s\.]+?)\s{2,}(.*?)\s{2,}(\w+\s+\d{1,2},?\s+\d{4})',
                line,
                re.IGNORECASE
            )
            
            if match:
                name = clean_name(match.group(1))
                title = clean_title(match.group(2))
                date = match.group(3).strip()
                
                # Check if title continues on next line (in parentheses)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('(') and next_line.endswith(')'):
                        title += ' ' + next_line
                        i += 1
                
                classification = classify_role(title)
                
                person = Person(
                    name=name,
                    title=title,
                    date=date,
                    signature=True,
                    **classification
                )
                people.append(person)
            
            else:
                # Try simpler pattern without date
                match2 = re.search(
                    r'/s/\s*([A-Za-z\s\.]+?)(?:\s{2,}(.*))?$',
                    line,
                    re.IGNORECASE
                )
                if match2:
                    name = clean_name(match2.group(1))
                    title = clean_title(match2.group(2)) if match2.group(2) else ""
                    
                    # Look ahead for title on next line
                    if not title and i + 1 < len(lines):
                        potential_title = lines[i + 1].strip()
                        if potential_title and not potential_title.startswith('/s/'):
                            title = clean_title(potential_title)
                            i += 1
                    
                    classification = classify_role(title)
                    
                    person = Person(
                        name=name,
                        title=title,
                        signature=True,
                        **classification
                    )
                    people.append(person)
        
        i += 1
    
    return people


def extract_people_from_table(text: str) -> List[Person]:
    """
    Extract people from table-style format.
    
    Example:
        Name                   Title                      Date
        John Doe              CEO                        Jan 1, 2025
        Jane Smith            CFO                        Jan 1, 2025
    """
    people = []
    
    lines = text.split('\n')
    
    # Look for table headers
    header_patterns = [
        r'name\s+title',
        r'name\s+position',
        r'director\s+title',
    ]
    
    table_started = False
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Detect table start
        if not table_started:
            if any(re.search(pattern, line_lower) for pattern in header_patterns):
                table_started = True
            continue
        
        # Skip separator lines
        if re.match(r'^[\s\-\_═]+$', line):
            continue
        
        # Try to extract name, title, and date from line
        # Pattern: NAME (2+ spaces) TITLE (2+ spaces) DATE
        match = re.search(
            r'^([A-Za-z\s\.]+?)\s{2,}(.*?)\s{2,}(\w+\s+\d{1,2},?\s+\d{4})',
            line
        )
        
        if match:
            name = clean_name(match.group(1))
            title = clean_title(match.group(2))
            date = match.group(3).strip()
            
            classification = classify_role(title)
            
            person = Person(
                name=name,
                title=title,
                date=date,
                signature=False,
                **classification
            )
            people.append(person)
    
    return people


def extract_people_from_list(text: str) -> List[Person]:
    """
    Extract people from list format.
    
    Example:
        John Doe, Chief Executive Officer
        Jane Smith, Chief Financial Officer
    """
    people = []
    
    # Pattern: NAME, TITLE
    pattern = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),\s*(.+?)(?:\n|$)'
    
    matches = re.finditer(pattern, text, re.MULTILINE)
    
    for match in matches:
        name = clean_name(match.group(1))
        title = clean_title(match.group(2))
        
        classification = classify_role(title)
        
        person = Person(
            name=name,
            title=title,
            signature=False,
            **classification
        )
        people.append(person)
    
    return people


def extract_people(text: str) -> List[Person]:
    """
    Main function to extract people from text using multiple strategies.
    
    Args:
        text: Text from SEC 10-K filing (ideally Item 10 or signature section)
    
    Returns:
        List of Person objects
    """
    # Try signature block format first (most common)
    people = extract_people_from_signature_block(text)
    
    # If no results, try table format
    if not people:
        people = extract_people_from_table(text)
    
    # If still no results, try list format
    if not people:
        people = extract_people_from_list(text)
    
    # Remove duplicates based on name
    seen_names = set()
    unique_people = []
    
    for person in people:
        name_normalized = person.name.lower().strip()
        if name_normalized and name_normalized not in seen_names:
            seen_names.add(name_normalized)
            unique_people.append(person)
    
    return unique_people


def format_people_output(people: List[Person], ticker: str = "") -> str:
    """
    Format list of people for human-readable output.
    
    Args:
        people: List of Person objects
        ticker: Optional company ticker
    
    Returns:
        Formatted string
    """
    if not people:
        return "No directors or officers found."
    
    header = f"**{ticker} - Directors and Officers**\n\n" if ticker else "**Directors and Officers**\n\n"
    
    # Group by role
    executives = [p for p in people if p.is_executive]
    directors = [p for p in people if p.is_director and not p.is_executive]
    other_officers = [p for p in people if p.is_officer and not p.is_executive and not p.is_director]
    
    output = header
    
    if executives:
        output += "**Executive Officers:**\n"
        for person in executives:
            output += f"• {person.name} - {person.title}\n"
        output += "\n"
    
    if directors:
        output += "**Directors:**\n"
        for person in directors:
            output += f"• {person.name} - {person.title}\n"
        output += "\n"
    
    if other_officers:
        output += "**Other Officers:**\n"
        for person in other_officers:
            output += f"• {person.name} - {person.title}\n"
        output += "\n"
    
    output += f"\n**Total:** {len(people)} people"
    
    return output.strip()
