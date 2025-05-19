"""
Strategy factory and registry for DynastAI simulation.
"""
from .random import RandomStrategy
from .balanced import BalancedStrategy
from .focus import (
    PietyFocusStrategy,
    StabilityFocusStrategy,
    PowerFocusStrategy,
    WealthFocusStrategy
)
from .llm_strategies.base import LLMStrategy
from .llm_strategies.conservative import ConservativeStrategy
from .llm_strategies.aggressive import AggressiveStrategy
from .llm_strategies.religious import ReligiousStrategy
from .llm_strategies.stability_first import PeopleFirstStrategy

# Strategy registry
_STRATEGIES = {
    'random': RandomStrategy,
    'balanced': BalancedStrategy,
    'piety_focus': PietyFocusStrategy,
    'stability_focus': StabilityFocusStrategy,
    'power_focus': PowerFocusStrategy,
    'wealth_focus': WealthFocusStrategy,
    'openrouter': LLMStrategy,
    'conservative': ConservativeStrategy,
    'aggressive': AggressiveStrategy,
    'religious': ReligiousStrategy,
    'stability_first': PeopleFirstStrategy
}

def get_strategy_names():
    """
    Get a list of all available strategy names
    
    Returns:
    --------
    list
        List of strategy names
    """
    return list(_STRATEGIES.keys())

def create_strategy(strategy_name, **kwargs):
    """
    Create a strategy instance by name
    
    Parameters:
    -----------
    strategy_name : str
        Name of the strategy to create
    **kwargs : dict
        Additional arguments to pass to the strategy constructor
        
    Returns:
    --------
    BaseStrategy
        Strategy instance
        
    Raises:
    -------
    ValueError
        If strategy_name is not recognized
    """
    if strategy_name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {', '.join(get_strategy_names())}")
    
    strategy_class = _STRATEGIES[strategy_name]
    return strategy_class(**kwargs)

# Export all available strategies
__all__ = [
    'RandomStrategy',
    'BalancedStrategy',
    'PietyFocusStrategy',
    'StabilityFocusStrategy',
    'PowerFocusStrategy',
    'WealthFocusStrategy',
    'LLMStrategy',
    'ConservativeStrategy',
    'AggressiveStrategy',
    'ReligiousStrategy',
    'PeopleFirstStrategy',
    'get_strategy_names',
    'create_strategy'
]