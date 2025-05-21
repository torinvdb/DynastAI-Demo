# DynastAI

## Overview

> Note: This is a Demo version of DynastAI which was created as an early POC. The new version will be located at [DynastAI](https://github.com/torinvdb/DynastAI).

DynastAI is a narrative kingdom management simulation inspired by the popular game Reigns. As the ruler of a medieval kingdom, you'll face a series of decisions presented by various characters from your realm. Each choice you make will shape your kingdom's future and affect your relationship with four crucial metrics: Piety, Stability, Power, and Wealth.

The game presents an engaging storytelling experience where your decisions have meaningful consequences. Will you be remembered as a benevolent ruler, a tyrant, or perhaps something in between? Your legacy is in your hands.

This repository contains both the core game simulation and an AI strategy analysis framework that allows you to compare different decision-making strategies, including LLM-based agents.

## Game Goal

Your primary objective is to maintain a delicate balance of power between the four metrics of your kingdom:
- **Piety**: Your relationship with religious authorities
- **Stability**: The common folk's satisfaction with your rule
- **Power**: The strength and loyalty of your armed forces
- **Wealth**: The financial health of your kingdom

Each metric is tracked on a scale from 0 to 100. Your reign continues as long as all four metrics remain within this range. If any metric falls to 0 or rises to 100, your rule will end - whether through religious uprising, popular revolt, military coup, or financial collapse.

The challenge lies in making decisions that satisfy some factions without completely alienating others. No ruler can please everyone, and the game tests your ability to navigate complex political waters while maintaining overall stability.

## Card Structure

The game operates using cards defined in the `cards/cards.csv` file. Each card represents a scenario requiring your decision. The cards are structured as follows:

* `ID`: Unique identifier for the card
* `Character`: The person presenting the decision (e.g., Diplomat, Merchant, General)
* `Prompt`: The scenario description and question posed to you
* `Left_Choice`: Text for the left swipe/reject option
* `Left_Piety`: How the left choice affects Piety (positive or negative integer)
* `Left_Stability`: How the left choice affects Stability
* `Left_Power`: How the left choice affects Power
* `Left_Wealth`: How the left choice affects Wealth
* `Right_Choice`: Text for the right swipe/accept option
* `Right_Piety`: How the right choice affects Piety
* `Right_Stability`: How the right choice affects Stability
* `Right_Power`: How the right choice affects Power
* `Right_Wealth`: How the right choice affects Wealth

## How to Play

1. **Start Your Reign**: Begin with balanced stats for all four kingdom metrics (typically 50 points each).

2. **Face Decisions**: Each turn, you'll be presented with a card featuring a character who brings a situation requiring your judgment.

3. **Make Choices**: For each card, you have two options:
   - Swipe left (or choose the left option) to reject/decline
   - Swipe right (or choose the right option) to accept/approve

4. **Watch Your Kingdom Change**: After each decision, your four kingdom metrics will change according to the values associated with your choice.

5. **Continue Your Dynasty**: Keep making decisions and try to maintain balance in your kingdom for as long as possible.

6. **End of Reign**: Your game ends when any metric reaches 0 or 100. The cause of your downfall will be related to whichever metric reached its limit.

## Example Card

Here's an example of how a card works in the game:

```
ID: 001
Character: Diplomat
Prompt: "With a sly smile, the diplomat gestures broadly: 'Sire, the lords quarrel like children. Shall we mediate disputes between lords?'"
Left Choice: "We cannot risk the kingdom's future; dismiss them with a royal wave."
Left Effects: Piety +10, Stability -10, Power +0, Wealth +0
Right Choice: "Make it so; our enemies shall kneel in terror!"
Right Effects: Piety -10, Stability +10, Power +0, Wealth +0
```

In this scenario, choosing the left option (dismissing the diplomat) would increase your Piety but decrease your Stability with the people. Choosing the right option (agreeing to mediate) would have the opposite effect.

## Strategy Tips

- **Balance is key**: Try to keep all metrics near the middle of their ranges to give yourself more flexibility.
- **Consider the long game**: Sometimes accepting a negative impact in one area is necessary for the overall health of your kingdom.
- **Watch for patterns**: Certain types of characters may consistently affect particular metrics of your kingdom.
- **Remember your history**: Your previous decisions may influence what scenarios appear later.

## Running DynastAI

### Prerequisites

1. **Python Environment**: Python 3.7+ is required
2. **Required Packages**: Install the necessary dependencies with:
   ```bash
   pip install -e .
   ```
   
### Running the Game
   
To launch the interactive game:
```bash
python main.py
```

### Running Simulations

To analyze different strategies with multiple simulation runs:
```bash
python -m simulation.main --num_sims 100 --strategies random balanced --output_dir results
```

Common options:
- `--num_sims`: Number of simulations per strategy (default: 500)
- `--max_turns`: Maximum turns per game (default: 200)
- `--strategies`: List of strategies to analyze (options: random, balanced, piety_focus, stability_focus, power_focus, wealth_focus)
- `--cards_file`: Path to custom cards CSV file
- `--output_dir`: Directory for saving analysis results

### Using LLM-based Strategies

The simulation supports using language models for decision-making:
```bash
python -m simulation.main --strategies openrouter --openrouter_model "openai/gpt-4-turbo" --num_sims 50
```

This requires an OpenRouter API key set in your `.env` file:
```
OPENROUTER_API_KEY=your_api_key_here
```

### Fine-Tuning Models

For fine-tuning language models to play DynastAI, see the instructions in the [fine-tuning directory](/fine-tuning/README.md).

## Enjoy Your Reign!

Whether you rule for generations or meet a swift end, DynastAI offers a unique storytelling experience where every decision matters. Good luck, Your Majesty!
