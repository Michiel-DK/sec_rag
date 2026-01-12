"""
ReAct agent for exploring company similarities.
Uses clean, simple tool definitions for reliability.
"""

import logging
import time
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


from sec_rag.similarity.similarity_engine import SimilarityEngine, load_similarity_engine
from sec_rag.llm.similarity_explainer import SimilarityExplainer, create_explainer
from .tools import create_agent_tools
from .prompts import AGENT_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityAgent:
    """
    Conversational agent for exploring company similarities.
    Uses ReAct pattern to decide which tools to call.
    """
    
    def __init__(
        self,
        similarity_engine: Optional[SimilarityEngine] = None,
        explainer: Optional[SimilarityExplainer] = None,
        model: str = "gemini-2.5-flash",
        verbose: bool = True,
        max_iterations: int = 15,  # INCREASED for complex queries
        max_retries: int = 3  # NEW
    ):
        """
        Initialize the agent.
        
        Args:
            similarity_engine: Similarity search engine (loads default if None)
            explainer: LLM explainer (creates default if None)
            model: LLM model to use for agent reasoning
            verbose: Whether to show agent reasoning steps
            max_iterations: Maximum reasoning iterations (default 10)
            max_retries: Maximum retries on API errors (default 3)
        """
        # Load components
        if similarity_engine is None:
            logger.info("Loading similarity engine...")
            similarity_engine = load_similarity_engine()
        
        if explainer is None:
            logger.info("Creating explainer...")
            explainer = create_explainer(similarity_engine.vector_store)
        
        self.similarity_engine = similarity_engine
        self.explainer = explainer
        self.max_retries = max_retries
        
        # Initialize LLM with retry configuration
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,
            max_retries=max_retries
        )
        
        # Create tools
        self.tools = create_agent_tools(similarity_engine, explainer)
        
        # Create simple ReAct prompt
        template = AGENT_SYSTEM_PROMPT + """

CONVERSATION HISTORY:
{chat_history}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: consider what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Add conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            output_key="output",
            input_key="input"
        )
        
        # Create executor with memory
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Force verbose to see agent trace
            handle_parsing_errors="Check your output and make sure it conforms to the format instructions!",
            max_iterations=20,  # Increased for multi-step queries
            max_execution_time=90,
            return_intermediate_steps=True,  # Return steps for debugging
            memory=self.memory
        )
        
        logger.info("✓ Similarity agent initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Tools: {len(self.tools)}")
        logger.info(f"  Max iterations: {max_iterations}")
    
    def run(self, query: str) -> str:
        """
        Run the agent with a user query, with retry logic.
        
        Args:
            query: Natural language question
        
        Returns:
            Agent's response
        """
        for attempt in range(self.max_retries):
            try:
                result = self.agent_executor.invoke({"input": query})
                
                # Debug: print intermediate steps if available
                if "intermediate_steps" in result:
                    print("\n=== AGENT TRACE ===")
                    for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                        print(f"\nStep {i}:")
                        print(f"  Action: {action.tool}")
                        print(f"  Input: {action.tool_input}")
                        print(f"  Output: {str(observation)[:100]}...")
                    print("===================\n")
                
                return result["output"]
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit / overload error
                if "503" in error_str or "overloaded" in error_str.lower():
                    if attempt < self.max_retries - 1:
                        wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                        logger.warning(f"⚠️  API overloaded. Retrying in {wait_time}s... (attempt {attempt + 2}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "The AI service is currently overloaded. Please try again in a few moments."
                
                # Check if agent hit iteration limit
                elif "iteration limit" in error_str.lower() or "time limit" in error_str.lower():
                    return "I couldn't complete the analysis in time. Try asking a more specific question, like focusing on one dimension (e.g., 'risk_profile' only)."
                
                # Other errors
                else:
                    logger.error(f"Agent error: {e}")
                    return f"I encountered an error: {str(e)[:200]}\nPlease try rephrasing your question or ask something simpler."
        
        return "Failed after multiple retries. Please try again later."
    
    def chat(self):
        """
        Interactive chat mode.
        """
        print("\n" + "=" * 60)
        print("🤖 Company Similarity Agent")
        print("=" * 60)
        print("\nAsk me about company similarities!")
        print("\nExample questions:")
        print('  • "Which companies are similar to Tesla?"')
        print('  • "Find tech companies similar to GOOG in risk_profile"')
        print('  • "Why are Apple and Microsoft similar?"')
        print('  • "Show me tech companies"')
        print('  • "Compare payment companies"')
        print("\nTips:")
        print("  - Be specific about dimensions (business_model, risk_profile, etc.)")
        print("  - Ask focused questions for best results")
        print("  - I remember our conversation, so you can ask follow-up questions!")
        print("\nType 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Debug: show memory on special command
                if query.lower() == 'show history':
                    print(f"\n📝 Chat History:\n{self.memory.load_memory_variables({})}\n")
                    continue
                
                if not query:
                    continue
                
                print("\n🤖 Agent: ", end="", flush=True)
                response = self.run(query)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue


def create_agent(
    model: str = "gemini-2.5-flash",
    verbose: bool = False,
    max_iterations: int = 15
) -> SimilarityAgent:
    """
    Factory function to create a similarity agent.
    
    Args:
        model: LLM model for agent reasoning
        verbose: Show reasoning steps
        max_iterations: Maximum iterations before stopping
    
    Returns:
        Configured SimilarityAgent
    """
    return SimilarityAgent(
        model=model,
        verbose=verbose,
        max_iterations=max_iterations
    )