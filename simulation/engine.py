"""
Core game engine for DynastAI simulation.
"""
import pandas as pd
import logging
import numpy as np
from .cards import get_card_value, check_card_dependencies

# Configure logging
logger = logging.getLogger(__name__)

class DynastAIGame:
    """
    Core game engine for the DynastAI simulation.
    
    This class manages the game state, including metrics, turns, and history.
    It uses a strategy object to make decisions.
    """
    
    def __init__(self, cards_df, strategy, starting_values=None):
        """
        Initialize a game simulation
        
        Parameters:
        -----------
        cards_df : pandas DataFrame
            DataFrame containing all card data
        strategy : BaseStrategy
            Strategy object to use for decision-making
        starting_values : dict or None
            Initial values for Power, Piety, Stability, Wealth
            Default is 50 for each
        """
        self.cards = cards_df
        self.strategy = strategy
        self.history = []
        self.turn = 0
        self.played_cards = set()  # Track played cards by ID
        
        # Set starting values (default is 50 for each)
        if starting_values is None:
            starting_values = {
                'Power': 50,
                'Piety': 50,
                'Stability': 50,
                'Wealth': 50
            }
        
        self.metrics = starting_values.copy()
        self.game_over = False
        self.end_reason = None
    
    def make_decision(self, card, choice):
        """
        Apply effects of a decision
        
        Parameters:
        -----------
        card : pandas Series
            The card being played
        choice : str
            Either 'Left' or 'Right'
        
        Returns:
        --------
        dict
            Updated metrics
        """
        # Apply the effects of the choice
        # This mapping is kept for backward compatibility with legacy cards
        # that might still use the old metric names
        metric_mapping = {
            'Crown': 'Power',
            'Church': 'Piety',
            'People': 'Stability',
            'Treasury': 'Wealth'
        }
        
        for new_metric in ['Power', 'Piety', 'Stability', 'Wealth']:
            try:
                # First try to get the effect using the new metric name
                effect = get_card_value(card, f'{choice}_{new_metric}')
            except KeyError:
                # If not found, try the legacy names using the reverse mapping
                legacy_mapping = {v: k for k, v in metric_mapping.items()}
                old_metric = legacy_mapping[new_metric]
                try:
                    effect = get_card_value(card, f'{choice}_{old_metric}')
                except KeyError:
                    # Try alternative column format
                    try:
                        effect = get_card_value(card, f'{choice} {old_metric}')
                    except KeyError:
                        # Try the new metric with a space
                        effect = get_card_value(card, f'{choice} {new_metric}')
            
            # Apply the effect to the metric
            self.metrics[new_metric] += effect
        
        # Record this turn's state
        card_id = get_card_value(card, 'ID')
        character = get_card_value(card, 'Character')
        
        # Add to played cards
        self.played_cards.add(card_id)
        
        self.history.append({
            'Turn': self.turn,
            'Card_ID': card_id,
            'Character': character,
            'Choice': choice,
            'Power': self.metrics['Power'],
            'Piety': self.metrics['Piety'],
            'Stability': self.metrics['Stability'],
            'Wealth': self.metrics['Wealth']
        })
        
        # Check for game over conditions
        for metric, value in self.metrics.items():
            if value <= 0:
                self.game_over = True
                self.end_reason = f"{metric} too low"
                # Only log in verbose mode or when explicitly needed for debugging
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logger.info(f"Game over: {self.end_reason} (survived {self.turn} turns)")
                break
            elif value >= 100:
                self.game_over = True
                self.end_reason = f"{metric} too high"
                # Only log in verbose mode or when explicitly needed for debugging
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logger.info(f"Game over: {self.end_reason} (survived {self.turn} turns)")
                break
        
        # Check for chain card to prioritize next
        try:
            chain_card_id = get_card_value(card, 'Chain_Card')
            if chain_card_id and pd.notna(chain_card_id) and chain_card_id not in self.played_cards:
                self.next_card_id = chain_card_id
            else:
                self.next_card_id = None
        except (KeyError, AttributeError):
            self.next_card_id = None
        
        return self.metrics.copy()
    
    def get_eligible_cards(self):
        """
        Get cards that are eligible to be played based on dependencies
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing eligible cards
        """
        if hasattr(self, 'next_card_id') and self.next_card_id:
            # If we have a chain card to play next, only return that card
            chain_card = self.cards[self.cards['ID'] == self.next_card_id]
            if not chain_card.empty:
                return chain_card
        
        # Filter cards that meet dependency requirements
        eligible_cards = []
        
        for _, card in self.cards.iterrows():
            # Skip cards that have already been played
            if get_card_value(card, 'ID') in self.played_cards:
                continue
                
            # Check dependencies
            if check_card_dependencies(card, self.history, self.metrics):
                eligible_cards.append(card)
        
        # If no cards are eligible (which shouldn't happen), fall back to all cards
        if not eligible_cards:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logger.warning("No eligible cards found, falling back to all unplayed cards")
            eligible_cards = [card for _, card in self.cards.iterrows() 
                             if get_card_value(card, 'ID') not in self.played_cards]
            
            # If literally all cards have been played, reset and allow replays
            if not eligible_cards:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logger.warning("All cards have been played, allowing replays")
                eligible_cards = [card for _, card in self.cards.iterrows()]
        
        return pd.DataFrame(eligible_cards)
    
    def play_turn(self):
        """
        Play a single turn using the strategy
        
        Returns:
        --------
        bool
            True if game continues, False if game over
        """
        if self.game_over:
            return False
        
        # Get eligible cards based on dependencies
        eligible_cards = self.get_eligible_cards()
        
        # Draw a random card from eligible cards
        card = eligible_cards.sample(1).iloc[0]
        
        # Let the strategy decide
        choice = self.strategy.make_decision(card, self.metrics, self.history)
        
        # Apply decision and increment turn counter
        self.make_decision(card, choice)
        self.turn += 1
        
        return not self.game_over
    
    def play_game(self, max_turns=200, verbose=False):
        """
        Play a complete game until game over or max_turns reached
        
        Parameters:
        -----------
        max_turns : int
            Maximum number of turns to play
        verbose : bool
            Whether to enable detailed logging
        
        Returns:
        --------
        dict
            Game statistics
        """
        if verbose:
            logger.info(f"Starting game with {self.strategy.name} strategy")
        
        while self.turn < max_turns and not self.game_over:
            self.play_turn()
        
        if not self.game_over and verbose:
            logger.info(f"Game reached max turns ({max_turns}) without ending")
        
        # Compile game statistics
        game_stats = {
            'strategy': self.strategy.name,
            'turns_survived': self.turn,
            'game_over': self.game_over,
            'end_reason': self.end_reason,
            'final_metrics': self.metrics.copy(),
            'history': pd.DataFrame(self.history)
        }
        
        return game_stats