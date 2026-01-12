"""
Test the similarity agent.
"""

import logging
from sec_rag.agent.similarity_agent import create_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_queries():
    """Test the agent with various queries."""
    
    print("=" * 80)
    print("AGENT TEST")
    print("=" * 80)
    
    # Create agent with verbose=True to see reasoning
    agent = create_agent(verbose=True, max_iterations=10)
    
    # Test queries
    test_queries = [
        "Which companies are similar to Tesla in risk_profile?",
        "Show me tech companies",
        "Why are Apple and Microsoft similar?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print(f"{'='*80}\n")
        
        response = agent.run(query)
        print(f"\nResponse: {response}\n")


def test_interactive():
    """Test interactive chat mode."""
    agent = create_agent(verbose=False, max_iterations=10)
    agent.chat()


if __name__ == "__main__":
    # Run interactive mode
    test_interactive()
    
    # Or run tests
    # test_agent_queries()