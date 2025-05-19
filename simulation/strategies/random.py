"""
Random strategy implementation for DynastAI simulation.
"""
import random
from .base import BaseStrategy

class RandomStrategy(BaseStrategy):
    """
    Strategy that makes completely random decisions.
    
    This strategy serves as both a baseline and a fallback when other strategies fail.
    """
    
    def make_decision(self, card, metrics, history):
        """
        Make a random decision
        
        Parameters:
        -----------
        card : dict or pandas Series
            The card being played
        metrics : dict
            Current game metrics (Piety, Stability, Power, Wealth)
        history : list
            List of previous turns
            
        Returns:
        --------
        str
            Either 'Left' or 'Right'
        """
        return random.choice(['Left', 'Right'])