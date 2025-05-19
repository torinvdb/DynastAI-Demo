"""
Card loading and validation module for DynastAI simulation.
"""
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_cards(filepath):
    """
    Load and process the cards data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing card data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing validated card data
    """
    try:
        # Check if the specified filepath exists
        if not os.path.exists(filepath):
            # If not, use the default location
            default_location = 'cards/cards.csv'
            if os.path.exists(default_location):
                filepath = default_location
                logger.info(f"Using card file from {default_location}")
            else:
                # Try the original default
                original_default = 'cards/cards.csv'
                if os.path.exists(original_default):
                    filepath = original_default
                    logger.info(f"Using card file from {original_default}")
            # Otherwise, we'll proceed with the original path and let it fail naturally
        
        cards_df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(cards_df)} cards from {filepath}")
        
        # Basic validation
        required_columns = [
            'ID', 'Character', 'Prompt', 
            'Left_Choice', 'Left_Piety', 'Left_Stability', 'Left_Power', 'Left_Wealth',
            'Right_Choice', 'Right_Piety', 'Right_Stability', 'Right_Power', 'Right_Wealth'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in cards_df.columns]
        if missing_columns:
            # Try alternative column names (with spaces instead of underscores)
            alt_missing = []
            for col in missing_columns:
                alt_col = col.replace('_', ' ')
                if alt_col in cards_df.columns:
                    # Rename to standardized format
                    cards_df = cards_df.rename(columns={alt_col: col})
                else:
                    alt_missing.append(col)
            
            if alt_missing:
                logger.warning(f"Missing required columns in card data: {alt_missing}")
                # We'll continue anyway but log the warning
        
        # Standardize column names and data types
        for col in cards_df.columns:
            if col.endswith(('Piety', 'Stability', 'Power', 'Wealth')):
                cards_df[col] = pd.to_numeric(cards_df[col], errors='coerce').fillna(0).astype(int)
        
        # Initialize dependency-related columns if they don't exist
        dependency_columns = ['Requires_Card', 'Requires_Choice', 'Requires_Metric', 'Requires_Range', 'Chain_Card']
        for col in dependency_columns:
            if col not in cards_df.columns:
                cards_df[col] = None
                
        return cards_df
        
    except Exception as e:
        logger.error(f"Error loading cards from {filepath}: {e}")
        
        # Create a fallback sample if file not found
        logger.info("Creating sample data...")
        
        # Sample data structure based on a few cards
        cards_list = generate_sample_cards()
        cards_df = pd.DataFrame(cards_list)
        logger.info(f"Created sample data with {len(cards_df)} cards")
        
        return cards_df

def generate_sample_cards():
    """
    Generate a small set of sample cards for testing when no file is available
    
    Returns:
    --------
    list
        List of dictionaries containing sample card data
    """
    # Sample cards
    return [
        {
            'ID': '001',
            'Character': 'Chancellor',
            'Prompt': 'Your Majesty, the royal coffers are nearly empty! We must raise taxes.',
            'Left_Choice': 'Raise taxes on peasants',
            'Left_Piety': 0,
            'Left_Stability': -5,
            'Left_Power': 0,
            'Left_Wealth': 8,
            'Right_Choice': 'Tax the nobles instead',
            'Right_Piety': 0,
            'Right_Stability': 3,
            'Right_Power': -2,
            'Right_Wealth': 5,
            'Requires_Card': None,
            'Requires_Choice': None,
            'Requires_Metric': None,
            'Requires_Range': None,
            'Chain_Card': None
        },
        {
            'ID': '002',
            'Character': 'General',
            'Prompt': 'The army needs new weapons for the upcoming campaign.',
            'Left_Choice': 'Approve funding',
            'Left_Piety': 0,
            'Left_Stability': -2,
            'Left_Power': 8,
            'Left_Wealth': -6,
            'Right_Choice': 'Deny the request',
            'Right_Piety': 0,
            'Right_Stability': 2,
            'Right_Power': -5,
            'Right_Wealth': 0,
            'Requires_Card': None,
            'Requires_Choice': None,
            'Requires_Metric': None,
            'Requires_Range': None,
            'Chain_Card': None
        },
        {
            'ID': '003',
            'Character': 'Bishop',
            'Prompt': 'The cathedral needs repairs, Your Majesty.',
            'Left_Choice': 'Fund the repairs',
            'Left_Piety': 7,
            'Left_Stability': 2,
            'Left_Power': 0,
            'Left_Wealth': -5,
            'Right_Choice': 'The Church can pay for it',
            'Right_Piety': -6,
            'Right_Stability': 0,
            'Right_Power': 0,
            'Right_Wealth': 0,
            'Requires_Card': None,
            'Requires_Choice': None,
            'Requires_Metric': None,
            'Requires_Range': None,
            'Chain_Card': None
        }
    ]

def get_card_value(card, key):
    """
    Safely get a value from a card, handling both dict and pandas.Series objects
    and alternative key formats
    
    Parameters:
    -----------
    card : dict or pandas.Series
        Card data
    key : str
        Key to retrieve
        
    Returns:
    --------
    any
        Value associated with the key
        
    Raises:
    -------
    KeyError
        If neither the key nor an alternative version exists
    """
    if key in card:
        return card[key]
    
    # Try alternative key format (spaces instead of underscores)
    alt_key = key.replace('_', ' ')
    if alt_key in card:
        return card[alt_key]
    
    # If using pandas Series, try attribute access
    if hasattr(card, 'get'):
        value = card.get(key, None)
        if value is not None:
            return value
        value = card.get(alt_key, None)
        if value is not None:
            return value
    
    raise KeyError(f"Key '{key}' or '{alt_key}' not found in card: {card}")

def check_card_dependencies(card, game_history, metrics):
    """
    Check if a card's dependencies are satisfied based on game history and current metrics
    
    Parameters:
    -----------
    card : pandas.Series
        The card to check
    game_history : list
        List of dictionaries containing game history
    metrics : dict
        Current game metrics
        
    Returns:
    --------
    bool
        True if dependencies are satisfied, False otherwise
    """
    try:
        # Check if there's a required previous card
        requires_card = get_card_value(card, 'Requires_Card')
        if requires_card and pd.notna(requires_card):
            # Check if the required card exists in history
            required_card_played = any(turn.get('Card_ID') == requires_card for turn in game_history)
            if not required_card_played:
                return False
            
            # Check if specific choice was required
            requires_choice = get_card_value(card, 'Requires_Choice')
            if requires_choice and pd.notna(requires_choice):
                # Find the turn with the required card
                required_turn = next((turn for turn in game_history if turn.get('Card_ID') == requires_card), None)
                if not required_turn or required_turn.get('Choice') != requires_choice:
                    return False
        
        # Check metric requirements
        requires_metric = get_card_value(card, 'Requires_Metric')
        requires_range = get_card_value(card, 'Requires_Range')
        
        if requires_metric and pd.notna(requires_metric) and requires_range and pd.notna(requires_range):
            if requires_metric not in metrics:
                return False
                
            metric_value = metrics[requires_metric]
            
            # Parse range requirement
            if '<' in requires_range:
                threshold = int(requires_range.replace('<', ''))
                if metric_value >= threshold:
                    return False
            elif '>' in requires_range:
                threshold = int(requires_range.replace('>', ''))
                if metric_value <= threshold:
                    return False
            
        return True
    
    except (KeyError, ValueError) as e:
        logger.warning(f"Error checking card dependencies for {card.get('ID')}: {e}")
        # On error, allow the card to be played
        return True