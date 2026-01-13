"""
Quick test for the get_directors_officers tool.
"""

from sec_rag.similarity.similarity_engine import load_similarity_engine
from sec_rag.llm.similarity_explainer import SimilarityExplainer
from sec_rag.agent.tools import create_agent_tools

def test_directors_tool():
    print("Testing get_directors_officers tool...\n")
    
    # Initialize components
    print("1. Loading vector store...")
    engine = load_similarity_engine(persist_dir="./chroma_db")
    
    print("2. Creating explainer...")
    explainer = SimilarityExplainer()
    
    print("3. Creating tools...")
    tools = create_agent_tools(engine, explainer)
    
    # Find the directors/officers tool
    directors_tool = None
    for tool in tools:
        if tool.name == "get_directors_officers":
            directors_tool = tool
            break
    
    if not directors_tool:
        print("ERROR: get_directors_officers tool not found!")
        return
    
    print(f"4. Found tool: {directors_tool.name}")
    print(f"   Description: {directors_tool.description[:100]}...")
    
    # Test with Apple
    print("\n5. Testing with AAPL...\n")
    print("=" * 80)
    result = directors_tool.func("ticker='AAPL'")
    print(result)
    print("=" * 80)

if __name__ == "__main__":
    test_directors_tool()
