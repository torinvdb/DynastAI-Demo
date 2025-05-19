"""
Analysis module for DynastAI simulation results.
"""
import numpy as np
import pandas as pd
import sys
import os
from collections import Counter, defaultdict
import logging
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# OutputFilter specific to this module
class OutputFilter:
    """Context manager to filter out certain outputs when not in verbose mode"""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.original_stdout = None
        self.null_output = None
        
    def __enter__(self):
        if not self.verbose:
            self.original_stdout = sys.stdout
            self.null_output = open(os.devnull, 'w')
            sys.stdout = self.null_output
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout = self.original_stdout
            self.null_output.close()

def quiet_unless_verbose(func):
    """
    Decorator to silence function output unless in verbose mode.
    Requires a 'verbose' parameter to be passed to the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get('verbose', False)
        with OutputFilter(verbose=verbose):
            return func(*args, **kwargs)
    return wrapper

@quiet_unless_verbose
def analyze_simulations(simulation_results, verbose=False):
    """
    Analyze the results of multiple simulations
    
    Parameters:
    -----------
    simulation_results : list
        List of game statistics from simulations
    verbose : bool, optional
        Whether to display verbose output
    
    Returns:
    --------
    dict
        Analysis results
    """
    if not simulation_results:
        logger.warning("No simulation results to analyze")
        return None
    
    # Extract key statistics
    turns_survived = [result['turns_survived'] for result in simulation_results]
    end_reasons = [result['end_reason'] for result in simulation_results]
    strategy = simulation_results[0]['strategy']  # Assume all have same strategy
    
    logger.info(f"Analyzing {len(simulation_results)} simulations for {strategy} strategy")
    
    # Calculate basic statistics
    basic_stats = {
        'count': len(turns_survived),
        'mean_turns': np.mean(turns_survived),
        'median_turns': np.median(turns_survived),
        'variance': np.var(turns_survived),
        'std_dev': np.std(turns_survived),
        'max_turns': np.max(turns_survived),
        'min_turns': np.min(turns_survived)
    }
    
    # Calculate survival probabilities
    max_observed_turn = max(turns_survived)
    survival_probs = {}
    for t in range(1, max_observed_turn + 1):
        surviving = sum(1 for turns in turns_survived if turns >= t)
        survival_probs[t] = surviving / len(turns_survived)
    
    # Analyze end reasons
    end_reason_counts = Counter(end_reasons)
    
    # Analyze state distributions at various points
    # Collect metrics at 10%, 25%, 50%, 75%, and 90% of each game's duration
    state_distributions = {
        'early': defaultdict(list),  # ~10% of game
        'quarter': defaultdict(list),  # ~25% of game
        'mid': defaultdict(list),  # ~50% of game
        'late': defaultdict(list),  # ~75% of game
        'final': defaultdict(list)   # End of game
    }
    
    for result in simulation_results:
        if 'history' in result and not result['history'].empty:
            history = result['history']
            game_length = len(history)
            
            if game_length > 0:
                # Calculate indices for different game stages
                early_idx = max(0, min(int(game_length * 0.1), game_length - 1))
                quarter_idx = max(0, min(int(game_length * 0.25), game_length - 1))
                mid_idx = max(0, min(int(game_length * 0.5), game_length - 1))
                late_idx = max(0, min(int(game_length * 0.75), game_length - 1))
                final_idx = game_length - 1
                
                # Collect metrics at each stage
                for metric in ['Piety', 'Stability', 'Power', 'Wealth']:
                    state_distributions['early'][metric].append(history.iloc[early_idx][metric])
                    state_distributions['quarter'][metric].append(history.iloc[quarter_idx][metric])
                    state_distributions['mid'][metric].append(history.iloc[mid_idx][metric])
                    state_distributions['late'][metric].append(history.iloc[late_idx][metric])
                    state_distributions['final'][metric].append(history.iloc[final_idx][metric])
    
    # Calculate card risk factors - which cards were played immediately before game over
    risky_cards = []
    for result in simulation_results:
        if result['game_over'] and 'history' in result and not result['history'].empty:
            history = result['history']
            if len(history) > 1:  # At least two turns played
                last_turn = history.iloc[-1]
                risky_cards.append((last_turn['Card_ID'], last_turn['Character']))
    
    card_risk = Counter(risky_cards)
    
    # Analyze metric trajectories
    metric_trajectories = {
        'Piety': defaultdict(list),
        'Stability': defaultdict(list),
        'Power': defaultdict(list),
        'Wealth': defaultdict(list)
    }
    
    # For each turn t, calculate the average value of each metric across all simulations
    # that reached at least t turns
    max_turn = max(turns_survived)
    for turn in range(max_turn + 1):
        for result in simulation_results:
            history = result['history']
            if len(history) > turn:
                turn_data = history.iloc[turn]
                for metric in metric_trajectories.keys():
                    metric_trajectories[metric][turn].append(turn_data[metric])
    
    # Calculate statistics for metric trajectories
    metric_trajectory_stats = {}
    for metric, turn_values in metric_trajectories.items():
        metric_trajectory_stats[metric] = {
            'turns': [],
            'mean': [],
            'median': [],
            'std_dev': [],
            'min': [],
            'max': [],
            'count': []
        }
        
        for turn, values in turn_values.items():
            if values:  # Only include turns with data
                metric_trajectory_stats[metric]['turns'].append(turn)
                metric_trajectory_stats[metric]['mean'].append(np.mean(values))
                metric_trajectory_stats[metric]['median'].append(np.median(values))
                metric_trajectory_stats[metric]['std_dev'].append(np.std(values))
                metric_trajectory_stats[metric]['min'].append(np.min(values))
                metric_trajectory_stats[metric]['max'].append(np.max(values))
                metric_trajectory_stats[metric]['count'].append(len(values))
    
    # Compile and return analysis results
    analysis = {
        'strategy': strategy,
        'basic_stats': basic_stats,
        'survival_probability': survival_probs,
        'end_reasons': dict(end_reason_counts),
        'state_distributions': state_distributions,
        'risky_cards': dict(card_risk),
        'metric_trajectories': metric_trajectory_stats
    }
    
    return analysis

@quiet_unless_verbose
def compare_strategies(strategy_results, verbose=False):
    """
    Compare multiple strategy analysis results
    
    Parameters:
    -----------
    strategy_results : dict
        Dictionary mapping strategy names to analysis results
    verbose : bool, optional
        Whether to display verbose output
    
    Returns:
    --------
    dict
        Comparison results
    """
    if not strategy_results:
        logger.warning("No strategy results to compare")
        return None
    
    comparison = {
        'strategies': list(strategy_results.keys()),
        'mean_turns': {},
        'median_turns': {},
        'max_turns': {},
        'survival_rates': {},
        'end_reasons': {},
        'metric_stability': {}
    }
    
    # Extract and compare key metrics
    for strategy, analysis in strategy_results.items():
        if not analysis:
            logger.warning(f"No analysis results for {strategy}")
            continue
            
        # Basic turn statistics
        comparison['mean_turns'][strategy] = analysis['basic_stats']['mean_turns']
        comparison['median_turns'][strategy] = analysis['basic_stats']['median_turns']
        comparison['max_turns'][strategy] = analysis['basic_stats']['max_turns']
        
        # Survival rates at key turn thresholds (25, 50, 100)
        comparison['survival_rates'][strategy] = {
            'turn_25': analysis['survival_probability'].get(25, 0),
            'turn_50': analysis['survival_probability'].get(50, 0),
            'turn_100': analysis['survival_probability'].get(100, 0)
        }
        
        # Most common end reasons
        comparison['end_reasons'][strategy] = dict(
            sorted(analysis['end_reasons'].items(), key=lambda x: x[1], reverse=True)[:3]
        )
        
        # Metric stability (standard deviation in mid-game)
        metric_stability = {}
        for metric in ['Piety', 'Stability', 'Power', 'Wealth']:
            mid_values = analysis['state_distributions']['mid'][metric]
            if mid_values:
                metric_stability[metric] = np.std(mid_values)
            else:
                metric_stability[metric] = None
        comparison['metric_stability'][strategy] = metric_stability
    
    # Determine the overall rankings
    if comparison['strategies']:
        # Rank by mean turns survived
        mean_turns_ranking = sorted(
            comparison['strategies'],
            key=lambda s: comparison['mean_turns'].get(s, 0),
            reverse=True
        )
        
        # Rank by survival rate at turn 50
        survival_ranking = sorted(
            comparison['strategies'],
            key=lambda s: comparison['survival_rates'].get(s, {}).get('turn_50', 0),
            reverse=True
        )
        
        # Add rankings to comparison
        comparison['rankings'] = {
            'by_mean_turns': mean_turns_ranking,
            'by_survival_rate': survival_ranking
        }
    
    return comparison

@quiet_unless_verbose
def get_card_impact_analysis(cards_df, verbose=False):
    """
    Analyze the potential impact of each card
    
    Parameters:
    -----------
    cards_df : pandas DataFrame
        DataFrame containing card data
    verbose : bool, optional
        Whether to display verbose output
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with card impact analysis
    """
    # Calculate impact magnitude for each card
    impact_data = []
    
    for _, card in cards_df.iterrows():
        # Get card identifier
        card_id = card['ID'] if 'ID' in card else card.get('id', 'unknown')
        character = card['Character'] if 'Character' in card else card.get('character', 'unknown')
        
        # Calculate impact for each metric
        metrics = ['Piety', 'Stability', 'Power', 'Wealth']
        
        left_impacts = {}
        right_impacts = {}
        for metric in metrics:
            try:
                left_impacts[metric] = card[f'Left_{metric}'] if f'Left_{metric}' in card else card[f'Left {metric}']
                right_impacts[metric] = card[f'Right_{metric}'] if f'Right_{metric}' in card else card[f'Right {metric}']
            except (KeyError, TypeError):
                logger.warning(f"Could not find impact values for card {card_id} and metric {metric}")
                left_impacts[metric] = 0
                right_impacts[metric] = 0
        
        # Calculate total impact magnitude
        left_total = sum(abs(impact) for impact in left_impacts.values())
        right_total = sum(abs(impact) for impact in right_impacts.values())
        max_total = max(left_total, right_total)
        
        # Calculate metric with maximum impact
        left_max_metric = max(left_impacts.items(), key=lambda x: abs(x[1]))[0]
        right_max_metric = max(right_impacts.items(), key=lambda x: abs(x[1]))[0]
        
        # Calculate potential imbalance (difference between metrics)
        left_imbalance = max(abs(left_impacts[m1] - left_impacts[m2]) 
                            for i, m1 in enumerate(metrics) 
                            for m2 in metrics[i+1:])
        right_imbalance = max(abs(right_impacts[m1] - right_impacts[m2]) 
                             for i, m1 in enumerate(metrics) 
                             for m2 in metrics[i+1:])
        
        impact_data.append({
            'Card_ID': card_id,
            'Character': character,
            'Left_Total_Impact': left_total,
            'Right_Total_Impact': right_total,
            'Max_Total_Impact': max_total,
            'Left_Max_Metric': left_max_metric,
            'Right_Max_Metric': right_max_metric,
            'Left_Max_Impact': max(abs(impact) for impact in left_impacts.values()),
            'Right_Max_Impact': max(abs(impact) for impact in right_impacts.values()),
            'Left_Imbalance': left_imbalance,
            'Right_Imbalance': right_imbalance,
            'Max_Imbalance': max(left_imbalance, right_imbalance)
        })
    
    return pd.DataFrame(impact_data).sort_values('Max_Total_Impact', ascending=False)