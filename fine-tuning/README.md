# DynastAI Fine-Tuning

This directory contains scripts and resources for fine-tuning language models to work with the DynastAI decision-making framework. The scripts train models to make balanced decisions that maximize reign length by maintaining the four key metrics (Piety, Stability, Power, and Wealth) in balanced states.

## Available Scripts

### Unsloth GRPO Fine-Tuning

`unsloth-grpo-reigns.py` - Uses the Unsloth library to fine-tune a model on the DynastAI card data with Gradient-based Return-Oriented Policy Optimization (GRPO). This approach provides accelerated training for Llama-based models.

### TLR GRPO Fine-Tuning

`tlr-grpo-reigns.py` - Uses TinyLlama-Reinforced approach for GRPO fine-tuning on DynastAI card data. This is optimized for smaller models with efficient training.

## Usage

Before running any fine-tuning scripts, ensure you have:

1. The required dependencies installed:
   ```bash
   pip install -r requirements.txt  # If provided
   # Or install dependencies manually:
   pip install unsloth transformers peft accelerate bitsandbytes
   ```

2. Appropriate GPU resources available (fine-tuning is computationally intensive)

3. The cards data available at `../cards/cards.csv`

### Running Fine-Tuning

To run the Unsloth GRPO fine-tuning:

```bash
python unsloth-grpo-reigns.py
```

To run the TLR GRPO fine-tuning:

```bash
python tlr-grpo-reigns.py
```

## Input Data Format

The fine-tuning scripts expect a card dataset with the following column structure:
- ID: Unique identifier
- Character: The character presenting the scenario
- Prompt: The scenario description
- Left_Choice: Text for the left option
- Left_Piety: Impact on Piety metric
- Left_Stability: Impact on Stability metric
- Left_Power: Impact on Power metric
- Left_Wealth: Impact on Wealth metric
- Right_Choice: Text for the right option
- Right_Piety: Impact on Piety metric
- Right_Stability: Impact on Stability metric
- Right_Power: Impact on Power metric
- Right_Wealth: Impact on Wealth metric

## Output

After fine-tuning, the models will be saved to the specified output directory. These models can be used with the DynastAI simulation to create AI agents that make decisions based on the card data they were trained on.
