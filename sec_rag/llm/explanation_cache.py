"""
Cache system for saving and loading LLM explanations.
Avoids re-running expensive API calls for previously generated explanations.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sec_rag.llm.types import SimilarityExplanation  # CHANGED: Import from types instead

class ExplanationCache:
    """
    Manages saving and loading of LLM-generated explanations.
    """
    
    def __init__(self, cache_dir: str = "./outputs/explanations"):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cached explanations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        
        logger.info(f"Explanation cache initialized at: {self.cache_dir}")
        logger.info(f"  Cached explanations: {len(self.index)}")
    
    def _load_index(self) -> Dict:
        """Load the index of cached explanations."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                return {}
        return {}
    
    def _save_index(self):
        """Save the index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _get_cache_key(self, ticker1: str, ticker2: str, dimension: str) -> str:
        """
        Generate a unique cache key for a comparison.
        Normalizes ticker order to avoid duplicates.
        """
        # Sort tickers to ensure AMZN vs MSFT = MSFT vs AMZN
        t1, t2 = sorted([ticker1.upper(), ticker2.upper()])
        return f"{t1}_vs_{t2}_{dimension}"
    
    def _get_cache_filename(self, cache_key: str) -> Path:
        """Get the filename for a cached explanation."""
        return self.cache_dir / f"{cache_key}.txt"
    
    def has_cached(self, ticker1: str, ticker2: str, dimension: str) -> bool:
        """
        Check if an explanation is already cached.
        
        Args:
            ticker1: First company ticker
            ticker2: Second company ticker
            dimension: Comparison dimension
        
        Returns:
            True if cached, False otherwise
        """
        cache_key = self._get_cache_key(ticker1, ticker2, dimension)
        return cache_key in self.index
    
    def get_cached(
        self,
        ticker1: str,
        ticker2: str,
        dimension: str
    ) -> Optional[SimilarityExplanation]:
        """
        Retrieve a cached explanation.
        
        Args:
            ticker1: First company ticker
            ticker2: Second company ticker
            dimension: Comparison dimension
        
        Returns:
            SimilarityExplanation if cached, None otherwise
        """
        cache_key = self._get_cache_key(ticker1, ticker2, dimension)
        
        if cache_key not in self.index:
            return None
        
        cache_file = self._get_cache_filename(cache_key)
        
        if not cache_file.exists():
            logger.warning(f"Cache file missing: {cache_file}")
            return None
        
        try:
            # Read the cached file
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse back into SimilarityExplanation
            metadata = self.index[cache_key]
            
            # Extract sections from the text file
            explanation = self._parse_cached_file(content)
            
            return SimilarityExplanation(
                ticker1=metadata['ticker1'],
                ticker2=metadata['ticker2'],
                dimension=metadata['dimension'],
                explanation=explanation['explanation'],
                key_similarities=explanation['similarities'],
                key_differences=explanation['differences'],
                confidence=explanation['confidence']
            )
            
        except Exception as e:
            logger.error(f"Failed to load cached explanation: {e}")
            return None
    
    def save(self, explanation: SimilarityExplanation) -> Path:
        """
        Save an explanation to the cache.
        
        Args:
            explanation: The SimilarityExplanation to cache
        
        Returns:
            Path to the saved file
        """
        cache_key = self._get_cache_key(
            explanation.ticker1,
            explanation.ticker2,
            explanation.dimension
        )
        cache_file = self._get_cache_filename(cache_key)
        
        # Format the explanation as human-readable text
        content = self._format_explanation(explanation)
        
        # Save to file
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update index
            self.index[cache_key] = {
                'ticker1': explanation.ticker1,
                'ticker2': explanation.ticker2,
                'dimension': explanation.dimension,
                'confidence': explanation.confidence,
                'cached_at': datetime.now().isoformat(),
                'filename': cache_file.name
            }
            
            self._save_index()
            
            logger.info(f"✓ Saved explanation to: {cache_file}")
            return cache_file
            
        except Exception as e:
            logger.error(f"Failed to save explanation: {e}")
            raise
    
    def _format_explanation(self, explanation: SimilarityExplanation) -> str:
        """
        Format an explanation as human-readable text.
        """
        lines = [
            "=" * 80,
            f"COMPANY SIMILARITY ANALYSIS",
            "=" * 80,
            "",
            f"Companies:    {explanation.ticker1} vs {explanation.ticker2}",
            f"Dimension:    {explanation.dimension.replace('_', ' ').title()}",
            f"Confidence:   {explanation.confidence.upper()}",
            f"Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=" * 80,
            "EXPLANATION",
            "=" * 80,
            "",
            explanation.explanation,
            "",
            "=" * 80,
            "KEY SIMILARITIES",
            "=" * 80,
            ""
        ]
        
        for i, similarity in enumerate(explanation.key_similarities, 1):
            lines.append(f"{i}. {similarity}")
        
        lines.extend([
            "",
            "=" * 80,
            "KEY DIFFERENCES",
            "=" * 80,
            ""
        ])
        
        for i, difference in enumerate(explanation.key_differences, 1):
            lines.append(f"{i}. {difference}")
        
        lines.extend([
            "",
            "=" * 80,
            "END OF ANALYSIS",
            "=" * 80,
        ])
        
        return "\n".join(lines)
    
    def _parse_cached_file(self, content: str) -> Dict:
        """
        Parse a cached text file back into structured data.
        """
        result = {
            'explanation': '',
            'similarities': [],
            'differences': [],
            'confidence': 'medium'
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line or '=' in line:
                continue
            
            # Detect sections
            if line == "EXPLANATION":
                current_section = 'explanation'
                continue
            elif line == "KEY SIMILARITIES":
                current_section = 'similarities'
                continue
            elif line == "KEY DIFFERENCES":
                current_section = 'differences'
                continue
            elif line.startswith("Confidence:"):
                confidence_text = line.split(':', 1)[1].strip().lower()
                if 'high' in confidence_text:
                    result['confidence'] = 'high'
                elif 'low' in confidence_text:
                    result['confidence'] = 'low'
                continue
            
            # Add content to current section
            if current_section == 'explanation' and not line.startswith('Companies:') and not line.startswith('Dimension:') and not line.startswith('Generated:'):
                result['explanation'] += line + ' '
            elif current_section == 'similarities' and line[0].isdigit():
                # Remove number prefix
                text = line.split('.', 1)[1].strip() if '.' in line else line
                result['similarities'].append(text)
            elif current_section == 'differences' and line[0].isdigit():
                text = line.split('.', 1)[1].strip() if '.' in line else line
                result['differences'].append(text)
        
        result['explanation'] = result['explanation'].strip()
        
        return result
    
    def list_cached(self) -> List[Dict]:
        """
        List all cached explanations.
        
        Returns:
            List of dictionaries with metadata about each cached explanation
        """
        return [
            {
                'key': key,
                **metadata
            }
            for key, metadata in self.index.items()
        ]
    
    def search_cache(
        self,
        ticker: Optional[str] = None,
        dimension: Optional[str] = None,
        confidence: Optional[str] = None
    ) -> List[Dict]:
        """
        Search the cache for specific explanations.
        
        Args:
            ticker: Filter by ticker (matches either ticker1 or ticker2)
            dimension: Filter by dimension
            confidence: Filter by confidence level
        
        Returns:
            List of matching cached explanations
        """
        results = []
        
        for key, metadata in self.index.items():
            # Apply filters
            if ticker and ticker.upper() not in [metadata['ticker1'], metadata['ticker2']]:
                continue
            
            if dimension and metadata['dimension'] != dimension:
                continue
            
            if confidence and metadata['confidence'] != confidence:
                continue
            
            results.append({
                'key': key,
                **metadata
            })
        
        return results
    
    def clear_cache(self, confirm: bool = False):
        """
        Clear all cached explanations.
        
        Args:
            confirm: Must be True to actually clear the cache
        """
        if not confirm:
            logger.warning("clear_cache() called without confirm=True. Not clearing cache.")
            return
        
        # Delete all cache files
        for cache_file in self.cache_dir.glob("*.txt"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete {cache_file}: {e}")
        
        # Clear index
        self.index = {}
        self._save_index()
        
        logger.info("✓ Cache cleared")