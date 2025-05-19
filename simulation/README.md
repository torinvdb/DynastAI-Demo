# DynastAI: A Simulation Framework ### Risk Function

The probability of reign ending on any given turn can be modeled as:

```
P(failure|s) = 1 - e^(-α · D(s)^β)
```

Where:
- `D(s)` is the weighted distance from the optimal center point (50,50,50,50) in the 4D space of Piety, Stability, Power, and Wealth
- `α` controls baseline risk (approximately 0.001 for balanced strategy)
- `β` controls risk acceleration (approximately 2, based on simulation)

This model explains why the balanced strategy dramatically outperforms others - by minimizing distance from the center point, it exponentially reduces failure risk.ased Strategy Games

DynastAI is a comprehensive framework for simulating and analyzing decision-based strategy games inspired by the popular game "Reigns". As the ruler of a medieval kingdom, you face a series of decisions presented by various characters from your realm. Each choice shapes your kingdom's future and affects your relationship with four crucial metrics: Piety, Stability, Power, and Wealth.

## Game Mechanics

### Core Gameplay

Players manage four critical kingdom metrics, each tracked on a scale from 0 to 100:
- **Piety**: Your relationship with religious authorities
- **Stability**: The common folk's satisfaction with your rule
- **Power**: The strength and loyalty of your armed forces
- **Wealth**: The financial health of your kingdom

Your reign continues as long as all four metrics remain within this range. If any metric falls to 0 or rises to 100, your rule ends through religious uprising, popular revolt, military coup, or financial collapse.

### Decision System

Each turn, you're presented with a card featuring a character who brings a situation requiring your judgment. You have two options:
- Left choice (typically representing a conservative/traditional approach)
- Right choice (typically representing a progressive/innovative approach)

Each choice impacts your four metrics differently, creating a complex balancing act where satisfying one faction often means alienating another.

## Mathematical Model

### The 4D State Space

DynastAI can be mathematically modeled as navigation through a four-dimensional state space:

1. Each metric (Piety, Stability, Power, Wealth) represents one dimension
2. Your game state at any moment is a point with coordinates (piety, stability, power, wealth) in this 4D space
3. Each decision moves you through this space by vector addition
4. Game over occurs when you hit any boundary (0 or 100) in any dimension

### Risk Function

The probability of reign ending on any given turn can be modeled as:

```
P(failure|s) = 1 - e^(-α · D(s)^β)
```

Where:
- `D(s)` is the weighted distance from the optimal center point (50,50,50,50) in the Piety, Stability, Power, Wealth state space
- `α` controls baseline risk (approximately 0.001 for balanced strategy)
- `β` controls risk acceleration (approximately 2, based on simulation)

This model explains why the balanced strategy dramatically outperforms others - by minimizing distance from the center point, it exponentially reduces failure risk.

## Framework Features

- **Multiple Strategy Types**: Test random, balanced, metric-focused, and even LLM-powered strategies
- **Parallel Simulation**: Run thousands of games efficiently with multiprocessing
- **Comprehensive Analysis**: Calculate statistics on survival rates, metric trajectories, game ending causes
- **Visualization Tools**: Generate insightful plots and comparisons between strategies
- **Extensible Design**: Easily add new strategies, cards, or analysis metrics

## Architecture

The codebase has been refactored into a modular design with clear separation of concerns:

```
simulation/
│
├── cards.py            # Card loading and validation
├── engine.py           # DynastAIGame class (core game logic)
├── simulation.py       # Simulation runners (single/multi, parallel)
├── analysis.py         # Analysis/statistics
├── visualization.py    # Plotting
│
├── strategies/
│   ├── __init__.py     # Strategy factory and registry
│   ├── base.py         # BaseStrategy class
│   ├── random.py       # RandomStrategy
│   ├── balanced.py     # BalancedStrategy
│   ├── focus.py        # Focused strategies
│   └── llm.py          # LLMStrategy (OpenRouter API)
```

## Installation

```bash
pip install -e .
```

## Usage

### Command Line Interface

Run a basic simulation with all available strategies:

```bash
dynastai --num_sims 100 --output_dir results
```

Specify particular strategies to test:

```bash
dynastai --strategies random balanced church_focus --num_sims 500
```

Using the OpenRouter LLM strategy:

```bash
dynastai --strategies openrouter --openrouter_model "openai/gpt-4-turbo"
```

Optional arguments:
- `--num_sims`: Number of simulations per strategy (default: 500)
- `--max_turns`: Maximum turns per game (default: 200)
- `--strategies`: List of specific strategies to analyze
- `--cards_file`: Path to custom cards CSV file
- `--output_dir`: Directory for saving analysis results

### Python API

```python
from simulation import load_cards, run_multi_strategy_simulations, analyze_simulations, create_visualizations

# Load card data
cards_df = load_cards('cards/cards.csv')

# Run simulations with multiple strategies
results = run_multi_strategy_simulations(
    cards_df,
    strategies=['random', 'balanced'],
    num_sims=100,
    max_turns=200
)

# Analyze the results
analyses = {}
for strategy, strategy_results in results.items():
    analyses[strategy] = analyze_simulations(strategy_results)

# Create visualizations
create_visualizations(analyses, cards_df, output_dir='results')
```

## Analysis Results

The framework generates comprehensive visualizations to help understand strategy performance:

### Survival Probability

The probability of survival dramatically varies by strategy, with the balanced approach providing orders of magnitude better performance.

### Average Reign Length

Balanced strategy achieves significantly longer reigns compared to all other approaches.

### End Reason Distribution

Different strategies lead to different failure modes, helping identify weaknesses in each approach.

### Metric Evolution

The distribution of metrics at different game stages reveals how various strategies maintain or fail to maintain balance.

### Risky Cards

Certain cards present significantly higher risk to your reign, allowing for tactical card-specific strategies.

### Card Impact Analysis

Cards with highest overall impact magnitude are identified to understand game dynamics.

## Key Findings

1. **Balanced Strategy Dominance**: The balanced strategy outperforms others by 2-3 orders of magnitude in reign length.

2. **Exponential Risk Model**: Game risk follows an exponential model where deviation from the center point dramatically increases failure probability.

3. **High Variance**: Even with optimal play, there's significant variability in reign length due to the stochastic nature of card draws.

4. **Focused Strategies Fail**: Single-metric focused strategies fail quickly by neglecting the other dimensions of rule.

5. **Game Difficulty**: The extreme performance difference between strategies indicates a challenging game where proper strategy matters enormously.

## Adding New Strategies

Create a new strategy by subclassing `BaseStrategy`:

```python
from simulation.strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def make_decision(self, card, metrics, history):
        # Your decision logic here
        return 'Left' if [some_condition] else 'Right'
```

Then register your strategy in `strategies/__init__.py`:

```python
from .my_strategy import MyCustomStrategy

# Add to strategy registry
_STRATEGIES['my_custom'] = MyCustomStrategy
```

## License

MIT