"""
Focused strategies that prioritize specific metrics.
"""
from .base import BaseStrategy
from simulation.cards import get_card_value

class FocusStrategy(BaseStrategy):
    """
    Base class for strategies that focus on a specific metric.
    """
    
    def __init__(self, focus_metric, low_threshold=40, high_threshold=60, target=50):
        """
        Initialize focus strategy
        
        Parameters:
        -----------
        focus_metric : str
            The metric to focus on (Piety, Stability, Power, or Wealth)
        low_threshold : int
            Threshold below which the metric is considered low
        high_threshold : int
            Threshold above which the metric is considered high
        target : int
            Target value for the metric when in the normal range
        """
        self.focus_metric = focus_metric
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.target = target
    
    @property
    def name(self):
        """Return the name of the strategy"""
        return f"{self.focus_metric.lower()}_focus"
    
    def make_decision(self, card, metrics, history):
        """
        Make a decision that prioritizes the focused metric
        
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
        # Get current metric value
        current = metrics[self.focus_metric]
        
        try:
            # Try getting the effect values with the standardized column names
            left_effect = get_card_value(card, f'Left_{self.focus_metric}')
            right_effect = get_card_value(card, f'Right_{self.focus_metric}')
        except KeyError:
            # Fallback to column names with spaces
            left_effect = get_card_value(card, f'Left {self.focus_metric}')
            right_effect = get_card_value(card, f'Right {self.focus_metric}')
        
        # Decision logic based on current metric value
        # If metric is low, want to increase it
        if current < self.low_threshold:
            choice = 'Left' if left_effect >= right_effect else 'Right'
        
        # If metric is high, want to decrease it
        elif current > self.high_threshold:
            choice = 'Left' if left_effect <= right_effect else 'Right'
        
        # If metric is in good range, choose balanced approach
        else:
            left_distance = abs((current + left_effect) - self.target)
            right_distance = abs((current + right_effect) - self.target)
            choice = 'Left' if left_distance <= right_distance else 'Right'
        
        return choice


class PietyFocusStrategy(FocusStrategy):
    """Strategy that prioritizes the Piety metric."""
    
    def __init__(self, **kwargs):
        super().__init__('Piety', **kwargs)


class StabilityFocusStrategy(FocusStrategy):
    """Strategy that prioritizes the Stability metric."""
    
    def __init__(self, **kwargs):
        super().__init__('Stability', **kwargs)


class PowerFocusStrategy(FocusStrategy):
    """Strategy that prioritizes the Power metric."""
    
    def __init__(self, **kwargs):
        super().__init__('Power', **kwargs)


class WealthFocusStrategy(FocusStrategy):
    """Strategy that prioritizes the Wealth metric."""
    
    def __init__(self, **kwargs):
        super().__init__('Wealth', **kwargs)