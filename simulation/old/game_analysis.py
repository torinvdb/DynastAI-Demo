import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from collections import Counter, defaultdict
import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import json
import concurrent.futures

# Set the aesthetic style of the plots
plt.style.use('fivethirtyeight')
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)

# Load the cards data
def load_cards(filepath):
    """
    Load and process the cards data from CSV file
    """
    try:
        cards_df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(cards_df)} cards")
        return cards_df
    except Exception as e:
        print(f"Error loading cards: {e}")
        # Create a fallback sample if file not found
        print("Creating sample data from the provided example...")
        
        # Sample data structure based on the first few cards in the game
        data = []
        with open('paste.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(('ID', '#', '\n')):  # Skip header rows or empty lines
                    continue
                data.append(line.strip())
        
        # Parse the CSV data
        cards_list = []
        for line in data:
            parts = line.split(',')
            if len(parts) >= 13:  # Ensure proper formatting
                card = {
                    'ID': parts[0],
                    'Character': parts[1],
                    'Prompt': parts[2].replace('"', ''),
                    'Left_Choice': parts[3].replace('"', ''),
                    'Left_Church': int(parts[4]),
                    'Left_People': int(parts[5]),
                    'Left_Military': int(parts[6]),
                    'Left_Treasury': int(parts[7]),
                    'Right_Choice': parts[8].replace('"', ''),
                    'Right_Church': int(parts[9]),
                    'Right_People': int(parts[10]),
                    'Right_Military': int(parts[11]),
                    'Right_Treasury': int(parts[12])
                }
                cards_list.append(card)
        
        cards_df = pd.DataFrame(cards_list)
        print(f"Created sample data with {len(cards_df)} cards")
        return cards_df

# --- OpenRouter LLM decision logic (concise, no class) ---
project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=project_root / '.env')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'YOUR_OPENROUTER_API_KEY')
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'

def get_card_value(card, key):
    if key in card:
        return card[key]
    alt_key = key.replace('_', ' ')
    if alt_key in card:
        return card[alt_key]
    raise KeyError(f"Key '{key}' or '{alt_key}' not found in card: {card}")

def openrouter_decision(card, metrics, history, system_prompt, model="google/gemini-2.5-flash-preview"):
    user_prompt = f"""
    You are playing a decision-based strategy game. Here is the current card:
    Character: {get_card_value(card, 'Character')}
    Prompt: {get_card_value(card, 'Prompt')}
    Left Choice: {get_card_value(card, 'Left_Choice')} (Church: {get_card_value(card, 'Left_Church')}, People: {get_card_value(card, 'Left_People')}, Military: {get_card_value(card, 'Left_Military')}, Treasury: {get_card_value(card, 'Left_Treasury')})
    Right Choice: {get_card_value(card, 'Right_Choice')} (Church: {get_card_value(card, 'Right_Church')}, People: {get_card_value(card, 'Right_People')}, Military: {get_card_value(card, 'Right_Military')}, Treasury: {get_card_value(card, 'Right_Treasury')})
    \nCurrent metrics:\nChurch: {metrics['Church']}, People: {metrics['People']}, Military: {metrics['Military']}, Treasury: {metrics['Treasury']}\n\nRespond with only 'Left' or 'Right' to indicate your decision. No explanation needed.\n"""
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1,
        "temperature": 0.0
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            reply = result['choices'][0]['message']['content'].strip()
            if reply.lower().startswith('left'):
                return 'Left'
            elif reply.lower().startswith('right'):
                return 'Right'
        print(f"OpenRouter API error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"OpenRouter API exception: {e}")
    return random.choice(['Left', 'Right'])

# Define the game class
class DynastAIGame:
    def __init__(self, cards_df, starting_values=None, openrouter_system_prompt=None, openrouter_model=None):
        """
        Initialize a game simulation
        
        Parameters:
        -----------
        cards_df : pandas DataFrame
            DataFrame containing all card data
        starting_values : dict
            Initial values for Church, People, Military, Treasury
            Default is 50 for each
        openrouter_system_prompt : str or None
            System prompt for OpenRouter LLM
        openrouter_model : str or None
            OpenRouter model name
        """
        self.cards = cards_df
        self.history = []
        self.turn = 0
        
        # Set starting values (default is 50 for each)
        if starting_values is None:
            starting_values = {
                'Church': 50,
                'People': 50,
                'Military': 50,
                'Treasury': 50
            }
        
        self.metrics = starting_values.copy()
        self.game_over = False
        self.end_reason = None
        self.openrouter_system_prompt = openrouter_system_prompt
        self.openrouter_model = openrouter_model
    
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
        if choice == 'Left':
            self.metrics['Church'] += card['Left Church']
            self.metrics['People'] += card['Left People']
            self.metrics['Military'] += card['Left Military']
            self.metrics['Treasury'] += card['Left Treasury']
        else:
            self.metrics['Church'] += card['Right Church']
            self.metrics['People'] += card['Right People']
            self.metrics['Military'] += card['Right Military']
            self.metrics['Treasury'] += card['Right Treasury']
        
        # Record this turn's state
        self.history.append({
            'Turn': self.turn,
            'Card_ID': card['ID'],
            'Character': card['Character'],
            'Choice': choice,
            'Church': self.metrics['Church'],
            'People': self.metrics['People'],
            'Military': self.metrics['Military'],
            'Treasury': self.metrics['Treasury']
        })
        
        # Check for game over conditions
        for metric, value in self.metrics.items():
            if value <= 0:
                self.game_over = True
                self.end_reason = f"{metric} too low"
                break
            elif value >= 100:
                self.game_over = True
                self.end_reason = f"{metric} too high"
                break
        
        return self.metrics.copy()
    
    def play_turn(self, strategy='random'):
        """
        Play a single turn with the given strategy
        
        Parameters:
        -----------
        strategy : str
            Game strategy. Options:
            - 'random': Random decisions
            - 'balanced': Try to keep all metrics balanced
            - 'church_focus': Prioritize Church metric
            - 'people_focus': Prioritize People metric
            - 'military_focus': Prioritize Military metric
            - 'treasury_focus': Prioritize Treasury metric
            - 'openrouter': Use OpenRouter strategy
        
        Returns:
        --------
        bool
            True if game continues, False if game over
        """
        if self.game_over:
            return False
        
        # Draw a random card
        card = self.cards.sample(1).iloc[0]
        
        # Determine the choice based on strategy
        if strategy == 'random':
            choice = random.choice(['Left', 'Right'])
        
        elif strategy == 'balanced':
            # Calculate which choice keeps metrics closer to center (50)
            left_imbalance = 0
            right_imbalance = 0
            
            for metric in ['Church', 'People', 'Military', 'Treasury']:
                current = self.metrics[metric]
                left_effect = card[f'Left {metric}']
                right_effect = card[f'Right {metric}']
                
                left_new = current + left_effect
                right_new = current + right_effect
                
                # Calculate distance from optimal (50)
                left_imbalance += abs(left_new - 50)
                right_imbalance += abs(right_new - 50)
            
            choice = 'Left' if left_imbalance <= right_imbalance else 'Right'
        
        elif '_focus' in strategy:
            # Extract the metric to focus on
            focus_metric = strategy.split('_')[0].capitalize()
            
            # Determine which choice is better for the focused metric
            current = self.metrics[focus_metric]
            left_effect = card[f'Left {focus_metric}']
            right_effect = card[f'Right {focus_metric}']
            
            # If metric is low, want to increase it
            if current < 40:
                choice = 'Left' if left_effect >= right_effect else 'Right'
            # If metric is high, want to decrease it
            elif current > 60:
                choice = 'Left' if left_effect <= right_effect else 'Right'
            # If metric is in good range, choose balanced approach
            else:
                target = 50
                left_distance = abs((current + left_effect) - target)
                right_distance = abs((current + right_effect) - target)
                choice = 'Left' if left_distance <= right_distance else 'Right'
        
        elif strategy == 'openrouter':
            card_dict = card.to_dict() if hasattr(card, 'to_dict') else card
            choice = openrouter_decision(
                card_dict,
                self.metrics.copy(),
                self.history.copy(),
                self.openrouter_system_prompt or (
                    "You are playing a decision-based kingdom management game. Each turn, you are presented with a scenario (card) and must choose between two options: 'Left' or 'Right'. "
                    "Each choice affects four metrics: Church, People, Military, and Treasury. "
                    "All metrics start at 50 and must always stay between 0 and 100. "
                    "If any metric goes below 1 or above 99, the game ends immediately. "
                    "Your objective is to survive as many turns as possible by making choices that keep all metrics within the safe range. "
                    "Respond with only 'Left' or 'Right' to indicate your decision. No explanation is needed."
                ),
                self.openrouter_model or "google/gemini-2.5-flash-preview"
            )
        
        else:
            # Default to random if strategy not recognized
            choice = random.choice(['Left', 'Right'])
        
        # Apply decision and increment turn counter
        self.make_decision(card, choice)
        self.turn += 1
        
        return not self.game_over
    
    def play_game(self, max_turns=200, strategy='random'):
        """
        Play a complete game until game over or max_turns reached
        
        Parameters:
        -----------
        max_turns : int
            Maximum number of turns to play
        strategy : str
            Game strategy to use
        
        Returns:
        --------
        dict
            Game statistics
        """
        while self.turn < max_turns and not self.game_over:
            self.play_turn(strategy)
        
        # Compile game statistics
        game_stats = {
            'turns_survived': self.turn,
            'game_over': self.game_over,
            'end_reason': self.end_reason,
            'final_metrics': self.metrics.copy(),
            'history': pd.DataFrame(self.history)
        }
        
        return game_stats

def run_single_simulation(cards_df, max_turns, strategy, openrouter_system_prompt, openrouter_model):
    game = DynastAIGame(cards_df, openrouter_system_prompt=openrouter_system_prompt, openrouter_model=openrouter_model)
    return game.play_game(max_turns, strategy)

def run_simulations(cards_df, num_sims=1000, max_turns=200, strategy='random', openrouter_system_prompt=None, openrouter_model=None, sim_workers=None, position=0):
    """
    Run multiple game simulations in parallel (per strategy)
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=sim_workers) as executor:
        futures = [
            executor.submit(
                run_single_simulation,
                cards_df,
                max_turns,
                strategy,
                openrouter_system_prompt,
                openrouter_model
            )
            for _ in range(num_sims)
        ]
        with tqdm(total=num_sims, desc=f"{strategy} simulations", position=position, leave=True) as pbar:
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
                pbar.update(1)
    return results

def run_and_analyze_strategy(strategy, cards_df, num_sims, max_turns, openrouter_system_prompt, openrouter_model, sim_workers, position):
    print(f"\nRunning simulations with {strategy} strategy...")
    results = run_simulations(
        cards_df,
        num_sims=num_sims,
        max_turns=max_turns,
        strategy=strategy,
        openrouter_system_prompt=openrouter_system_prompt,
        openrouter_model=openrouter_model,
        sim_workers=sim_workers,
        position=position
    )
    analysis = analyze_simulations(results)
    return strategy, analysis

def analyze_simulations(simulation_results):
    """
    Analyze the results of multiple simulations
    
    Parameters:
    -----------
    simulation_results : list
        List of game statistics from simulations
    
    Returns:
    --------
    dict
        Analysis results
    """
    # Extract key statistics
    turns_survived = [result['turns_survived'] for result in simulation_results]
    end_reasons = [result['end_reason'] for result in simulation_results]
    
    # Calculate metrics
    mean_turns = np.mean(turns_survived)
    median_turns = np.median(turns_survived)
    var_turns = np.var(turns_survived)
    std_turns = np.std(turns_survived)
    max_turns = np.max(turns_survived)
    min_turns = np.min(turns_survived)
    
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
                for metric in ['Church', 'People', 'Military', 'Treasury']:
                    state_distributions['early'][metric].append(history.iloc[early_idx][metric])
                    state_distributions['quarter'][metric].append(history.iloc[quarter_idx][metric])
                    state_distributions['mid'][metric].append(history.iloc[mid_idx][metric])
                    state_distributions['late'][metric].append(history.iloc[late_idx][metric])
                    state_distributions['final'][metric].append(history.iloc[final_idx][metric])
    
    # Calculate card risk factors
    # Identify which cards were played immediately before game over
    risky_cards = []
    for result in simulation_results:
        if result['game_over'] and 'history' in result and not result['history'].empty:
            history = result['history']
            if len(history) > 1:  # At least two turns played
                last_turn = history.iloc[-1]
                risky_cards.append((last_turn['Card_ID'], last_turn['Character']))
    
    card_risk = Counter(risky_cards)
    
    # Compile analysis results
    analysis = {
        'basic_stats': {
            'mean_turns': mean_turns,
            'median_turns': median_turns,
            'variance': var_turns,
            'std_dev': std_turns,
            'max_turns': max_turns,
            'min_turns': min_turns
        },
        'survival_probability': survival_probs,
        'end_reasons': end_reason_counts,
        'state_distributions': state_distributions,
        'risky_cards': card_risk
    }
    
    return analysis

# Visualization functions
def create_visualizations(analyses, strategies, cards_df, output_dir='.'):
    """
    Create visualizations of game analyses
    
    Parameters:
    -----------
    analyses : dict
        Dictionary of analysis results for each strategy
    strategies : list
        List of strategy names
    cards_df : pandas DataFrame
        Cards data
    output_dir : str
        Output directory for saving figures
    """
    # 1. Survival probability curves
    plt.figure(figsize=(12, 8))
    for strategy in strategies:
        survival_probs = analyses[strategy]['survival_probability']
        turns = list(survival_probs.keys())
        probs = list(survival_probs.values())
        plt.plot(turns, probs, label=strategy.replace('_', ' ').title())
    
    plt.xlabel('Turns')
    plt.ylabel('Probability of Survival')
    plt.title('Survival Probability by Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/survival_probability.png', dpi=300, bbox_inches='tight')
    
    # 2. Mean turns survived by strategy (bar chart)
    plt.figure(figsize=(10, 6))
    mean_turns = [analyses[strategy]['basic_stats']['mean_turns'] for strategy in strategies]
    std_devs = [analyses[strategy]['basic_stats']['std_dev'] for strategy in strategies]
    
    plt.bar(
        [s.replace('_', ' ').title() for s in strategies], 
        mean_turns,
        yerr=std_devs,
        capsize=5,
        alpha=0.7
    )
    plt.xlabel('Strategy')
    plt.ylabel('Mean Turns Survived')
    plt.title('Average Reign Length by Strategy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mean_turns.png', dpi=300, bbox_inches='tight')
    
    # 3. End reason distribution (pie charts)
    for strategy in strategies:
        plt.figure(figsize=(10, 8))
        end_reasons = analyses[strategy]['end_reasons']
        
        # Filter out None values if any
        if None in end_reasons:
            del end_reasons[None]
        
        labels = list(end_reasons.keys())
        sizes = list(end_reasons.values())
        
        plt.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            shadow=False, 
            startangle=90,
            textprops={'fontsize': 12}
        )
        plt.axis('equal')
        plt.title(f'End Reasons for {strategy.replace("_", " ").title()} Strategy')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/end_reasons_{strategy}.png', dpi=300, bbox_inches='tight')
    
    # 4. State distribution at different game stages (boxplots)
    metrics = ['Church', 'People', 'Military', 'Treasury']
    stages = ['early', 'quarter', 'mid', 'late', 'final']
    
    for strategy in strategies:
        state_dist = analyses[strategy]['state_distributions']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            data = [state_dist[stage][metric] for stage in stages]
            
            plt.boxplot(
                data,
                tick_labels=[s.capitalize() for s in stages],
                showfliers=False
            )
            plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Initial Value')
            plt.axhline(y=0, color='crimson', linestyle='-', alpha=0.5, label='Lower Limit')
            plt.axhline(y=100, color='crimson', linestyle='-', alpha=0.5, label='Upper Limit')
            
            plt.xlabel('Game Stage')
            plt.ylabel(f'{metric} Value')
            plt.title(f'{metric} Distribution Over Time ({strategy.replace("_", " ").title()})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/{strategy}_{metric}_distribution.png', dpi=300, bbox_inches='tight')
    
    # 5. Card risk analysis (top 10 riskiest cards for each strategy)
    for strategy in strategies:
        risky_cards = analyses[strategy]['risky_cards']
        if not risky_cards:
            continue
            
        top_risky = dict(risky_cards.most_common(10))
        
        plt.figure(figsize=(12, 8))
        card_ids = [f"{id} ({char})" for (id, char) in top_risky.keys()]
        counts = list(top_risky.values())
        
        plt.barh(card_ids, counts)
        plt.xlabel('Count of Game-Ending Appearances')
        plt.ylabel('Card ID (Character)')
        plt.title(f'Top 10 Riskiest Cards ({strategy.replace("_", " ").title()})')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{strategy}_risky_cards.png', dpi=300, bbox_inches='tight')
    
    # 6. Card impact analysis (calculate and visualize impact magnitude for each card)
    plt.figure(figsize=(14, 10))
    
    # Calculate total impact magnitude for each card
    impact_magnitudes = []
    for _, card in cards_df.iterrows():
        left_impact = abs(card['Left Church']) + abs(card['Left People']) + abs(card['Left Military']) + abs(card['Left Treasury'])
        right_impact = abs(card['Right Church']) + abs(card['Right People']) + abs(card['Right Military']) + abs(card['Right Treasury'])
        max_impact = max(left_impact, right_impact)
        
        impact_magnitudes.append({
            'Card_ID': card['ID'],
            'Character': card['Character'],
            'Impact': max_impact
        })
    
    impact_df = pd.DataFrame(impact_magnitudes)
    impact_df = impact_df.sort_values('Impact', ascending=False).head(20)
    
    plt.barh(
        [f"{row['Card_ID']} ({row['Character']})" for _, row in impact_df.iterrows()],
        impact_df['Impact']
    )
    plt.xlabel('Maximum Impact Magnitude')
    plt.ylabel('Card ID (Character)')
    plt.title('Top 20 Cards with Highest Impact')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/highest_impact_cards.png', dpi=300, bbox_inches='tight')

# Main function
def main(num_sims=500, max_turns=200, strategies=None, openrouter_system_prompt=None, openrouter_model=None, strategy_workers=None, sim_workers=10):
    """
    Main function to run the analysis
    """
    print("DynastAI Game Analysis")
    print("----------------------")
    
    # Load cards data
    try:
        cards_df = load_cards('../cards/cards.csv')
    except:
        cards_df = load_cards('paste.txt')
    
    # Card data exploration
    print("\nCard Data Statistics:")
    print(f"Total cards: {len(cards_df)}")
    
    # Display impact statistics for each metric
    metrics = ['Church', 'People', 'Military', 'Treasury']
    for metric in metrics:
        left_values = cards_df[f'Left {metric}']
        right_values = cards_df[f'Right {metric}']
        
        print(f"\n{metric} Impact Statistics:")
        print(f"  Left choice - Min: {left_values.min()}, Max: {left_values.max()}, Mean: {left_values.mean():.2f}")
        print(f"  Right choice - Min: {right_values.min()}, Max: {right_values.max()}, Mean: {right_values.mean():.2f}")
    
    # Define strategies to analyze
    if strategies is None:
        strategies = ['random', 'balanced', 'church_focus', 'people_focus', 'military_focus', 'treasury_focus', 'openrouter']
    
    # Run simulations for each strategy in parallel, with unique tqdm positions
    all_analyses = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=strategy_workers) as executor:
        futures = {
            executor.submit(
                run_and_analyze_strategy,
                strategy,
                cards_df,
                num_sims,
                max_turns,
                openrouter_system_prompt,
                openrouter_model,
                sim_workers,
                idx
            ): strategy for idx, strategy in enumerate(strategies)
        }
        for future in concurrent.futures.as_completed(futures):
            strategy, analysis = future.result()
            all_analyses[strategy] = analysis
            # Print basic statistics
            stats = analysis['basic_stats']
            print(f"\n{strategy.replace('_', ' ').title()} Strategy Results:")
            print(f"  Mean turns survived: {stats['mean_turns']:.2f}")
            print(f"  Median turns survived: {stats['median_turns']:.2f}")
            print(f"  Standard deviation: {stats['std_dev']:.2f}")
            print(f"  Min turns: {stats['min_turns']}")
            print(f"  Max turns: {stats['max_turns']}")
            # Print most common end reasons
            print("\n  Most common end reasons:")
            for reason, count in analysis['end_reasons'].most_common(3):
                if reason:
                    print(f"    {reason}: {count} games ({count/sum(analysis['end_reasons'].values())*100:.1f}%)")
            # Print top risky cards
            print("\n  Top 5 riskiest cards:")
            for (card_id, character), count in analysis['risky_cards'].most_common(5):
                print(f"    Card {card_id} ({character}): {count} games")
    
    # Create output directory for results
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(all_analyses, strategies, cards_df, output_dir=output_dir)
    
    print(f"\nAnalysis complete! Results saved in '{output_dir}' folder.")
    
    # Return the analyses for further exploration
    return all_analyses, cards_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DynastAI Game Analysis CLI")
    parser.add_argument('--num_sims', type=int, default=500, help='Number of simulations per strategy (default: 500)')
    parser.add_argument('--max_turns', type=int, default=200, help='Maximum turns per game (default: 200)')
    parser.add_argument('--strategies', nargs='+', default=None, help='List of strategies to analyze (default: all)')
    parser.add_argument('--openrouter_system_prompt', type=str, default=None, help='System prompt for OpenRouter LLM (default: expert survival)')
    parser.add_argument('--openrouter_model', type=str, default=None, help='OpenRouter model name (default: openai/gpt-3.5-turbo)')
    parser.add_argument('--strategy_workers', type=int, default=None, help='Number of parallel workers for strategies (default: all CPUs)')
    parser.add_argument('--sim_workers', type=int, default=10, help='Number of parallel workers for simulations per strategy (default: 10)')

    args = parser.parse_args()

    all_analyses, cards_df = main(
        num_sims=args.num_sims,
        max_turns=args.max_turns,
        strategies=args.strategies,
        openrouter_system_prompt=args.openrouter_system_prompt,
        openrouter_model=args.openrouter_model,
        strategy_workers=args.strategy_workers,
        sim_workers=args.sim_workers
    )