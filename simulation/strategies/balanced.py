"""
Balanced strategy implementation for DynastAI simulation.
"""
from .base import BaseStrategy
from ..cards import get_card_value

class BalancedStrategy(BaseStrategy):
    """
    Strategy that tries to keep all metrics balanced around the center (50).
    
    Makes decisions that minimize the deviation of all metrics from the optimal value.
    """
    
    def __init__(self, target_value=50):
        """
        Initialize balanced strategy
        
        Parameters:
        -----------
        target_value : int
            The optimal value to keep metrics close to (default: 50)
        """
        self.target_value = target_value
    
    def make_decision(self, card, metrics, history):
        """
        Make a decision that keeps metrics closer to the center
        
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
        # Calculate which choice keeps metrics closer to center (target_value)
        left_imbalance = 0
        right_imbalance = 0
        
        for metric in ['Piety', 'Stability', 'Power', 'Wealth']:
            current = metrics[metric]
            
            try:
                left_effect = get_card_value(card, f'Left_{metric}')
                right_effect = get_card_value(card, f'Right_{metric}')
            except KeyError:
                # Fallback to column names with spaces
                left_effect = get_card_value(card, f'Left {metric}')
                right_effect = get_card_value(card, f'Right {metric}')
            
            left_new = current + left_effect
            right_new = current + right_effect
            
            # Calculate distance from optimal (target_value)
            left_imbalance += abs(left_new - self.target_value)
            right_imbalance += abs(right_new - self.target_value)
        
        return 'Left' if left_imbalance <= right_imbalance else 'Right'