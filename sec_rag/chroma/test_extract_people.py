"""
Test the director/officer extraction functionality.
"""

from sec_rag.chroma.extract_people import extract_people, format_people_output

# Test with the sample data from the user
test_text = """
  Name                        Title                                               Date              
  /s/ Timothy D. Cook         Chief Executive Officer and Director                October 31, 2025  
                              (Principal Executive Officer)                                         
 ────────────────────────────────────────────────────────────────────────────────────────────────────
  TIMOTHY D. COOK                                                                                   
  /s/ Kevan Parekh            Senior Vice President, Chief Financial Officer      October 31, 2025  
  KEVAN PAREKH                                                                                      
  /s/ Chris Kondo             Senior Director of Corporate Accounting             October 31, 2025  
  CHRIS KONDO                                                                                       
  /s/ Wanda Austin            Director                                            October 31, 2025  
  WANDA AUSTIN                                                                                      
  /s/ Alex Gorsky             Director                                            October 31, 2025  
  ALEX GORSKY                                                                                       
  /s/ Andrea Jung             Director                                            October 31, 2025  
  ANDREA JUNG                                                                                       
  /s/ Arthur D. Levinson      Director and Chair of the Board                     October 31, 2025  
  ARTHUR D. LEVINSON                                                                                
  /s/ Monica Lozano           Director                                            October 31, 2025  
  MONICA LOZANO                                                                                     
  /s/ Ronald D. Sugar         Director                                            October 31, 2025  
  RONALD D. SUGAR                                                                                   
  /s/ Susan L. Wagner         Director                                            October 31, 2025  
  SUSAN L. WAGNER
"""

def test_extraction():
    print("Testing director/officer extraction...\n")
    print("=" * 80)
    
    # Extract people
    people = extract_people(test_text)
    
    print(f"\nExtracted {len(people)} people:\n")
    
    for person in people:
        print(f"Name: {person.name}")
        print(f"Title: {person.title}")
        print(f"Date: {person.date}")
        print(f"Executive: {person.is_executive}, Director: {person.is_director}, Officer: {person.is_officer}")
        print("-" * 80)
    
    print("\nFormatted output:\n")
    print(format_people_output(people, "AAPL"))

if __name__ == "__main__":
    test_extraction()
