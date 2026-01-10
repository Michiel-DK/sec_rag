"""
Query templates for finding company similarities across different dimensions.
"""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DimensionQuery:
    """A query for a specific dimension of company comparison."""
    dimension: str
    queries: List[str]
    sections: List[str]  # Which 10-K sections to search
    weight: float  # Importance weight (0-1)

class ComparisonDimensions:
    """
    Predefined queries for comparing companies across key dimensions.
    """
    
    @staticmethod
    def business_model() -> DimensionQuery:
        """Queries for business model similarity."""
        return DimensionQuery(
            dimension="business_model",
            queries=[
                "primary revenue sources and business segments",
                "main products and services offered to customers",
                "customer base and target markets",
                "distribution channels and sales strategy",
                "competitive advantages and differentiation",
            ],
            sections=["Item 1 - Business", "Item 7 - MD&A"],
            weight=0.30
        )
    
    @staticmethod
    def risk_profile() -> DimensionQuery:
        """Queries for risk profile similarity."""
        return DimensionQuery(
            dimension="risk_profile",
            queries=[
                "regulatory compliance and legal risks",
                "competitive landscape and market risks",
                "technology and cybersecurity risks",
                "operational and supply chain risks",
                "financial and liquidity risks",
            ],
            sections=["Item 1A - Risk Factors"],
            weight=0.25
        )
    
    @staticmethod
    def financial_structure() -> DimensionQuery:
        """Queries for financial structure similarity."""
        return DimensionQuery(
            dimension="financial_structure",
            queries=[
                "debt obligations and credit facilities",
                "revenue growth trends and profitability",
                "capital allocation and investment strategy",
                "cash flow generation and liquidity position",
                "dividend policy and share repurchase programs",
            ],
            sections=["Item 7 - MD&A", "Item 8 - Financial Statements"],
            weight=0.20
        )
    
    @staticmethod
    def geographic_footprint() -> DimensionQuery:
        """Queries for geographic similarity."""
        return DimensionQuery(
            dimension="geographic_footprint",
            queries=[
                "international operations and foreign markets",
                "revenue by geographic region",
                "global expansion strategy and plans",
                "foreign currency exposure and risks",
            ],
            sections=["Item 1 - Business", "Item 7 - MD&A"],
            weight=0.15
        )
    
    @staticmethod
    def legal_matters() -> DimensionQuery:
        """Queries for legal/regulatory similarity."""
        return DimensionQuery(
            dimension="legal_matters",
            queries=[
                "pending litigation and legal proceedings",
                "regulatory investigations and compliance matters",
                "intellectual property disputes",
                "class action lawsuits and settlements",
            ],
            sections=["Item 3 - Legal Proceedings", "Item 1A - Risk Factors"],
            weight=0.10
        )
    
    @classmethod
    def all_dimensions(cls) -> List[DimensionQuery]:
        """Get all comparison dimensions."""
        return [
            cls.business_model(),
            cls.risk_profile(),
            cls.financial_structure(),
            cls.geographic_footprint(),
            cls.legal_matters(),
        ]
    
    @classmethod
    def get_dimension(cls, name: str) -> DimensionQuery:
        """Get a specific dimension by name."""
        dimensions = {
            "business_model": cls.business_model(),
            "risk_profile": cls.risk_profile(),
            "financial_structure": cls.financial_structure(),
            "geographic_footprint": cls.geographic_footprint(),
            "legal_matters": cls.legal_matters(),
        }
        
        if name not in dimensions:
            raise ValueError(f"Unknown dimension: {name}. Choose from: {list(dimensions.keys())}")
        
        return dimensions[name]