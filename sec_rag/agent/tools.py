"""
Tools for the similarity agent.
Simple, clean tool definitions - NO WRAPPERS.
"""

import json
from typing import Optional
from langchain.tools import Tool

from sec_rag.similarity.similarity_engine import SimilarityEngine
from sec_rag.llm.similarity_explainer import SimilarityExplainer


def create_agent_tools(
    similarity_engine: SimilarityEngine,
    explainer: SimilarityExplainer
) -> list:
    """
    Create LangChain tools from existing functionality.
    Clean, simple functions with default parameters - no wrappers, no parsing.
    """
    
    # Tool 1: Find similar companies
    def find_similar_impl(ticker: str = "", dimension: str = "all", limit: int = 5) -> str:
        """Find companies similar to a target ticker."""
        try:
            if not ticker:
                return "Error: ticker is required (e.g., 'GOOG')"
            
            ticker = ticker.upper().strip()
            limit = min(int(limit), 5)
            
            # Handle dimension parameter
            dimensions = None if dimension == "all" else [dimension]
            
            rankings = similarity_engine.find_similar_companies(
                target_ticker=ticker,
                dimensions=dimensions,
                top_n=limit
            )
            
            if not rankings:
                return f"No similar companies found for {ticker}."
            
            # Concise output
            result = f"Companies similar to {ticker}:\n"
            for i, r in enumerate(rankings, 1):
                result += f"{i}. {r.ticker} (score: {r.overall_score:.2f})\n"
            
            return result.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def find_similar(input_str: str) -> str:
        """Wrapper that parses JSON or dict input."""
        try:
            # Handle dict input (from agent)
            if isinstance(input_str, dict):
                return find_similar_impl(**input_str)
            
            # Try to parse as JSON first
            try:
                params = json.loads(input_str)
            except:
                # If not JSON, try to parse as key=value format
                params = {}
                for part in input_str.split(','):
                    if '=' in part:
                        key, val = part.split('=', 1)
                        params[key.strip()] = val.strip().strip("'\"")
            
            return find_similar_impl(**params)
        except Exception as e:
            return f"Error parsing input: {str(e)}"
    
    # Tool 2: Explain similarity between two companies
    def explain_two_impl(ticker1: str = "", ticker2: str = "", dimension: str = "risk_profile") -> str:
        """Explain why two specific companies are similar."""
        try:
            # Handle missing arguments
            if not ticker1:
                return "Error: ticker1 is required (e.g., 'GOOG')"
            if not ticker2:
                return "Error: ticker2 is required (e.g., 'MSFT')"
            
            ticker1 = ticker1.upper().strip()
            ticker2 = ticker2.upper().strip()
            
            explanation = explainer.explain_similarity(
                ticker1=ticker1,
                ticker2=ticker2,
                dimension=dimension
            )
            
            result = f"**{explanation.ticker1} vs {explanation.ticker2}** ({dimension})\n\n"
            result += f"{explanation.explanation}\n\n"
            result += "Key similarities:\n"
            for sim in explanation.key_similarities[:2]:
                result += f"• {sim}\n"
            
            return result.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def explain_two(input_str: str) -> str:
        """Wrapper that parses JSON input."""
        try:
            # Try to parse as JSON first
            try:
                params = json.loads(input_str)
            except:
                # If not JSON, try to parse as key=value format
                params = {}
                for part in input_str.split(','):
                    if '=' in part:
                        key, val = part.split('=', 1)
                        params[key.strip()] = val.strip().strip("'\"")
            
            return explain_two_impl(**params)
        except Exception as e:
            return f"Error parsing input: {str(e)}"
    
    # Tool 3: Explain why multiple companies are similar to one target
    def explain_multiple_impl(ticker: str = "", others: str = "", dimension: str = "risk_profile") -> str:
        """Explain why several companies are similar to a target."""
        try:
            # Handle missing arguments
            if not ticker:
                return "Error: ticker is required (e.g., 'GOOG')"
            if not others:
                return "Error: others is required (comma-separated, e.g., 'MSFT,AAPL')"
            
            ticker = ticker.upper().strip()
            
            # Parse comma-separated tickers
            other_tickers = [t.strip().upper() for t in others.split(',')]
            other_tickers = other_tickers[:3]  # Limit to 3
            
            result = f"Why these companies are similar to {ticker} ({dimension}):\n\n"
            
            for i, other in enumerate(other_tickers, 1):
                try:
                    exp = explainer.explain_similarity(
                        ticker1=ticker,
                        ticker2=other,
                        dimension=dimension
                    )
                    result += f"{i}. **{exp.ticker2}**: {exp.explanation[:150]}...\n\n"
                except Exception as e:
                    result += f"{i}. **{other}**: Error - {str(e)[:50]}\n\n"
            
            return result.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def explain_multiple(input_str: str) -> str:
        """Wrapper that parses JSON input."""
        try:
            # Try to parse as JSON first
            try:
                params = json.loads(input_str)
            except:
                # If not JSON, try to parse as key=value format
                params = {}
                for part in input_str.split(','):
                    if '=' in part:
                        key, val = part.split('=', 1)
                        params[key.strip()] = val.strip().strip("'\"")
            
            return explain_multiple_impl(**params)
        except Exception as e:
            return f"Error parsing input: {str(e)}"
    
    # Tool 4: Retrieve specific information from SEC filings
    def get_info_impl(ticker: str = "", query: str = "") -> str:
        """Get specific information from a company's SEC 10-K filing."""
        try:
            # Handle missing arguments
            if not ticker:
                return "Error: ticker is required (e.g., 'GOOG')"
            if not query:
                return "Error: query is required (e.g., 'intellectual property')"
            
            ticker = ticker.upper().strip()
            
            docs = similarity_engine.vector_store.similarity_search(
                query,
                k=3,
                filter={"ticker": ticker}
            )
            
            if not docs:
                return f"No information found for {ticker} about '{query}'."
            
            result = f"**{ticker}** - {query}:\n\n"
            for i, doc in enumerate(docs, 1):
                section = doc.metadata.get('section', 'Unknown')
                content = doc.page_content[:250].strip()
                result += f"{i}. [{section}]\n{content}...\n\n"
            
            return result.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_info(input_str: str) -> str:
        """Wrapper that parses JSON input."""
        try:
            # Try to parse as JSON first
            try:
                params = json.loads(input_str)
            except:
                # If not JSON, try to parse as key=value format
                params = {}
                for part in input_str.split(','):
                    if '=' in part:
                        key, val = part.split('=', 1)
                        params[key.strip()] = val.strip().strip("'\"")
            
            return get_info_impl(**params)
        except Exception as e:
            return f"Error parsing input: {str(e)}"
    
    # Tool 5: List available companies
    def list_companies(industry: str = "all") -> str:
        """List available companies, optionally filtered by industry."""
        try:
            results = similarity_engine.vector_store.similarity_search("company", k=100)
            all_tickers = sorted(set(doc.metadata.get('ticker', '') for doc in results if doc.metadata.get('ticker')))
            
            industry_map = {
                'tech': ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'META', 'NVDA', 'AMD', 'TSLA', 'CSCO', 'ORCL'],
                'finance': ['MA', 'V', 'AXP', 'JPM', 'MS', 'BAC', 'WFC'],
                'healthcare': ['ABBV', 'ABT', 'JNJ', 'LLY', 'UNH'],
                'consumer': ['COST', 'WMT', 'HD', 'KO', 'PG'],
                'energy': ['CVX', 'XOM']
            }
            
            if industry.lower() != "all" and industry.lower() in industry_map:
                filtered = [t for t in all_tickers if t in industry_map[industry.lower()]]
                return f"{industry.title()} companies: {', '.join(filtered)}"
            
            return f"Available companies ({len(all_tickers)}): {', '.join(all_tickers)}"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create tools list
    return [
        Tool(
            name="find_similar_companies",
            func=find_similar,
            description="""Find companies similar to a target.
            
Args:
  ticker (str, required): Company ticker (e.g., 'GOOG')
  dimension (str, optional): Specific dimension to focus on (business_model, risk_profile, 
    financial_structure, geographic_footprint, legal_matters) or 'all' for all dimensions (default: 'all')
  limit (int, optional): Number of results, max 5 (default: 5)

Example: ticker='GOOG', dimension='risk_profile', limit=5"""
        ),
        
        Tool(
            name="explain_similarity",
            func=explain_two,
            description="""Explain why TWO specific companies are similar.
            
Args:
  ticker1 (str, required): First company ticker (e.g., 'GOOG')
  ticker2 (str, required): Second company ticker (e.g., 'MSFT')
  dimension (str, optional): Comparison dimension (default: 'risk_profile')
    Options: business_model, risk_profile, financial_structure, geographic_footprint, legal_matters

Example: ticker1='GOOG', ticker2='MSFT', dimension='risk_profile'"""
        ),
        
        Tool(
            name="explain_multiple_similarities",
            func=explain_multiple,
            description="""Explain why MULTIPLE companies are similar to one target.
            
Args:
  ticker (str, required): Target company (e.g., 'GOOG')
  others (str, required): Comma-separated list of companies (e.g., 'MA,MSFT,LLY')
  dimension (str, optional): Comparison dimension (default: 'risk_profile')

Example: ticker='GOOG', others='MA,MSFT,LLY', dimension='risk_profile'"""
        ),
        
        Tool(
            name="retrieve_company_information",
            func=get_info,
            description="""Get specific information from a company's SEC 10-K filing.
            
Args:
  ticker (str, required): Company ticker (e.g., 'GOOG')
  query (str, required): What to search for (e.g., 'intellectual property patents', 'key products')

Example: ticker='GOOG', query='intellectual property patents'"""
        ),
        
        Tool(
            name="list_companies",
            func=list_companies,
            description="""List available companies.
            
Args:
  industry (str, optional): Filter by industry (tech, finance, healthcare, consumer, energy) 
    or 'all' for all companies (default: 'all')

Example: industry='tech'"""
        ),
    ]
