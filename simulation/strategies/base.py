"""
Base strategy module for DynastAI simulation.
All strategies should inherit from the BaseStrategy class.
"""
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.
    
    A strategy is responsible for deciding which choice to make when presented with a card.
    """
    
    @property
    def name(self):
        """
        Return the name of the strategy
        """
        # Default to the class name, but can be overridden
        return self.__class__.__name__.replace('Strategy', '').lower()
    
    @abstractmethod
    def make_decision(self, card, metrics, history):
        """
        Make a decision based on the current game state and card
        
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
        pass