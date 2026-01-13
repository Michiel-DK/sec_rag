"""
Test the agent with director/officer queries.
"""

from sec_rag.agent.similarity_agent import create_agent

def test_directors_query():
    """Test the agent with director/officer extraction query."""
    
    print("=" * 80)
    print("TESTING DIRECTOR/OFFICER EXTRACTION IN AGENT")
    print("=" * 80)
    
    # Create agent
    print("\nCreating agent...")
    agent = create_agent(verbose=True, max_iterations=10)
    
    # Test queries
    test_queries = [
        "Who are the directors and officers of Apple?",
        "Show me the executive team at Microsoft",
        "Get directors and officers for TSLA",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print(f"{'='*80}\n")
        
        try:
            response = agent.run(query)
            print(f"\n✓ Response:\n{response}\n")
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    test_directors_query()
