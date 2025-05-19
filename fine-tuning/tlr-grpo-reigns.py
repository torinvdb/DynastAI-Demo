# %% [markdown]
# # DynastAI GRPO Training
# 
# This script trains a model to make balanced decisions in the DynastAI kingdom simulation game.
# The model learns to select options that keep all kingdom metrics (Piety, Stability, Power, Wealth)
# balanced close to 50.

# %% Import Libraries
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model # Added import
import huggingface_hub  # Ensure huggingface_hub is available
import logging # Added import
try:
    import hf_xet  # Optional: Only needed for Xet Storage acceleration
except ImportError:
    print("[INFO] For faster model downloads with Xet Storage, run: pip install huggingface_hub[hf_xet] or pip install hf_xet")

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %% [markdown]
# ## Data Loading and Formatting Functions
# 
# These functions handle loading card data and formatting it into prompts.

# %% Card Processing Functions
def load_cards_data(csv_path="cards/cards.csv"):
    """Load cards data from CSV file"""
    return pd.read_csv(csv_path)

def format_card_as_prompt(card, current_metrics):
    """Format a card as a user prompt with metrics and options"""
    # Format the current metrics
    metrics_text = "Current Metrics:\n" + "\n".join([f"{metric}: {value}" for metric, value in current_metrics.items()])
    
    # Format the option effects
    option1_effects = f"(Piety {card['Left_Piety']:+d}, Stability {card['Left_Stability']:+d}, Power {card['Left_Power']:+d}, Wealth {card['Left_Wealth']:+d})"
    option2_effects = f"(Piety {card['Right_Piety']:+d}, Stability {card['Right_Stability']:+d}, Power {card['Right_Power']:+d}, Wealth {card['Right_Wealth']:+d})"
    
    # Create the complete prompt
    prompt = f"{metrics_text}\n\n{card['Character']} says: \"{card['Prompt']}\"\n\nOptions:\n"
    prompt += f"1. {card['Left_Choice']} {option1_effects}\n"
    prompt += f"2. {card['Right_Choice']} {option2_effects}"
    prompt += " /no_think"  # Added /no_think to user prompt
    
    return prompt

def determine_optimal_choice(card, current_metrics, target=50):
    """Calculate which choice (1 or 2) better balances the metrics"""
    # Calculate current total imbalance (sum of distances from target)
    current_imbalance = sum(abs(value - target) for value in current_metrics.values())
    
    # Calculate imbalance after option 1 (Left)
    option1_metrics = current_metrics.copy()
    option1_metrics['Piety'] += card['Left_Piety']
    option1_metrics['Stability'] += card['Left_Stability']
    option1_metrics['Power'] += card['Left_Power']
    option1_metrics['Wealth'] += card['Left_Wealth']
    option1_imbalance = sum(abs(value - target) for value in option1_metrics.values())
    
    # Calculate imbalance after option 2 (Right)
    option2_metrics = current_metrics.copy()
    option2_metrics['Piety'] += card['Right_Piety']
    option2_metrics['Stability'] += card['Right_Stability']
    option2_metrics['Power'] += card['Right_Power']
    option2_metrics['Wealth'] += card['Right_Wealth']
    option2_imbalance = sum(abs(value - target) for value in option2_metrics.values())
    
    # Return option with less imbalance (better balance)
    return "1" if option1_imbalance <= option2_imbalance else "2"

# %% [markdown]
# ## Dataset Creation
# 
# This section generates training examples and formats them correctly for GRPO training.

# %% Dataset Creation
def create_dataset(cards_data, num_examples=500):
    """Create a training dataset with examples of decision scenarios"""
    system_prompt = "You are a royal advisor. Choose the option (1 or 2) that will best balance the kingdom's metrics (Piety, Stability, Power, Wealth) closest to 50. Respond with ONLY the option number - either 1 or 2. /no_think"
    
    examples = []
    for _ in range(num_examples):
        # Randomly select a card
        card = cards_data.sample(1).iloc[0]
        
        # Generate random metrics between 10 and 90
        current_metrics = {
            'Piety': np.random.randint(10, 90),
            'Stability': np.random.randint(10, 90),
            'Power': np.random.randint(10, 90),
            'Wealth': np.random.randint(10, 90)
        }
        
        # Format the user prompt and determine optimal choice
        user_prompt = format_card_as_prompt(card, current_metrics)
        optimal_choice = determine_optimal_choice(card, current_metrics)
        
        examples.append({
            "card": card.to_dict(),
            "metrics": current_metrics,
            "prompt": user_prompt,
            "solution": optimal_choice
        })
    
    # Convert to Huggingface Dataset
    dataset = Dataset.from_list(examples)
    
    # Format dataset as system/user prompts with answers
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["prompt"]},
        ],
        "answer": x["solution"],
    })
    
    return dataset

# %% [markdown]
# ## Reward Function
# 
# This function calculates rewards based on whether the model chose the optimal option.

# %% Reward Function
def reward_function(completions, prompts, answer, **kwargs):
    """Reward model outputs based on whether they match the optimal choice"""
    logger.debug(f"--- reward_function START ---")
    logger.debug(f"Batch size: {len(completions)}")
    if prompts and len(prompts) > 0:
        logger.debug(f"Sample prompt (first item): {prompts[0]}")
    else:
        logger.debug("Prompts list is empty or None.")
    if completions and len(completions) > 0:
        logger.debug(f"Sample completion (first item, raw): {completions[0]}")
    else:
        logger.debug("Completions list is empty or None.")
    if answer and len(answer) > 0:
        logger.debug(f"Sample correct answer (first item): {answer[0]}")
    else:
        logger.debug("Answer list is empty or None.")

    rewards = []
    
    for i, (completion_item, correct_answer) in enumerate(zip(completions, answer)):
        logger.debug(f"Processing item {i+1}/{len(completions)}:")
        logger.debug(f"  Correct answer: '{correct_answer}'")
        logger.debug(f"  Raw completion_item: {completion_item}")

        # Extract the model's choice (should be just "1" or "2")
        # Completion is a list of dicts, e.g., [{'role': 'assistant', 'content': ' 1'}]
        if not completion_item or not isinstance(completion_item, list) or len(completion_item) == 0 or "content" not in completion_item[0]:
            logger.warning(f"  Unexpected completion format for item {i}: {completion_item}. Assigning raw_model_output as empty string.")
            raw_model_output = ""
        else:
            raw_model_output = completion_item[0]["content"].strip()
        
        logger.debug(f"  Extracted raw_model_output: '{raw_model_output}'")
        
        # Find the first occurrence of "1" or "2" in the response
        model_choice = None
        for char_idx, char_val in enumerate(raw_model_output):
            if char_val in ["1", "2"]:
                model_choice = char_val
                logger.debug(f"  Valid model_choice '{model_choice}' found at char index {char_idx} in raw_model_output.")
                break
        
        if model_choice is None:
            logger.warning(f"  No valid choice ('1' or '2') found in raw_model_output: '{raw_model_output}'. Assigning default reward -1.0.")
            current_reward = -1.0
        elif model_choice == correct_answer:
            current_reward = 1.0  # Correct choice
            logger.debug(f"  Model choice '{model_choice}' MATCHES correct_answer '{correct_answer}'. Reward: {current_reward}")
        else:
            current_reward = -1.0  # Incorrect choice
            logger.debug(f"  Model choice '{model_choice}' MISMATCHES correct_answer '{correct_answer}'. Reward: {current_reward}")
            
        rewards.append(current_reward)
    
    logger.debug(f"Generated rewards for batch: {rewards}")
    logger.debug(f"--- reward_function END ---")
    return rewards


# %% Main Execution
# %%
# === 1. Load card data ===
csv_path = "/home/earlpotters/DynastAI/cards/cards.csv"
print(f"Loading card data from {csv_path}...")
cards_data = load_cards_data(csv_path)

# %%
# === 2. Create dataset ===
num_examples = 50
print(f"Creating dataset with {num_examples} examples...")
dataset = create_dataset(cards_data, num_examples)
print("Dataset example:")
print(dataset[0])

# %%
# === 3. Load model and tokenizer ===
model_name = "Qwen/Qwen3-1.7B"
print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === 3.1 Configure LoRA ===
print("Configuring LoRA...")
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"], # Common for Qwen models, adjust if needed
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
# === 4. Configure GRPO trainer ===
print("Configuring GRPO trainer...")
grpo_config = GRPOConfig(
    num_generations=4,             # Generate 4 different completions per prompt
    max_prompt_length=512,         # Maximum prompt length
    max_completion_length=64,       # Very short completions - just "1" or "2"
    learning_rate=1e-5,            # Learning rate (updated)
    lr_scheduler_type="linear",    # Added: Type of learning rate scheduler
    warmup_steps=0,                # Added: Number of warmup steps
    num_train_epochs=3,            # Number of training epochs
    per_device_train_batch_size=4, # Batch size per device
    gradient_accumulation_steps=2, # Gradient accumulation steps
    output_dir="dynast_ai_grpo_model", # Output directory
    temperature=0.7,               # Temperature for generation
    logging_steps=1,
    bf16=True, # Added based on example
    remove_unused_columns=False # Added based on example, to keep 'answer' and 'prompt' for reward
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer
)

# %%
# === 5. Train ===
print("Training started...")
trainer.train()

# %%
# === 6. Save model ===
print("Saving model...")
trainer.save_model("dynast_ai_grpo_final")
print("Training completed and model saved!")
