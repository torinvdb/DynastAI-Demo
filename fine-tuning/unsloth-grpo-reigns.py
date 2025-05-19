# %% [markdown]
# # DynastAI GRPO Training with Unsloth
# 
# This script trains a model using GRPO (Generative Reward-based Policy Optimization) to make balanced decisions in the DynastAI kingdom simulation game.
# The model learns to select options that keep all kingdom metrics (Piety, Stability, Power, Wealth) balanced close to 50.

# %% Import Libraries
from unsloth import FastLanguageModel
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
import re
import logging
import wandb
from tqdm import tqdm  # Add tqdm for progress bar
from huggingface_hub import HfApi, create_repo, upload_folder, login
import os

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log in to Weights & Biases (ensure your API key is set in the environment or use wandb.login(key="YOUR_KEY") if needed)
wandb.login()

# Initialize wandb
wandb.init(project="dynastai-grpo-reigns", name="unsloth-grpo-run")

# Authenticate with Hugging Face Hub
# Recommended: Set your token in the environment variable HUGGINGFACE_TOKEN
# Example: export HUGGINGFACE_TOKEN=your_hf_token
hf_token = os.environ.get("HUGGINGFACE_TOKEN", "<your_hf_token>")
if hf_token is None:
    print("HUGGINGFACE_TOKEN environment variable not found.")
    hf_token = input("Please enter your Hugging Face access token: ").strip()
    if not hf_token:
        raise ValueError("No Hugging Face token provided. Exiting.")
else:
    print("Using Hugging Face token from environment variable.")
login(token=hf_token)

# %% [markdown]
# ## Model Initialization
# 
# Initialize the Qwen 1.7B model with LoRA for fine-tuning

# %% Initialize Model and LoRA
max_seq_length = 4096  # Increased for much longer sequences
lora_rank = 16  # Smaller rank for 1.7B model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-1.7B",  # Using Qwen 1.7B as specified
    max_seq_length = max_seq_length,
    load_in_4bit = False,  # False for LoRA 16bit
    fast_inference = True,  # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,  # Reduce if out of memory
    # attention_implementation = "flash_attention_2", ## Unsloth claims to be faster
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2,  # *2 speeds up training
    use_gradient_checkpointing = "unsloth",  # Reduces memory usage
    random_state = 3407,
)

# %% [markdown]
# ## System Prompt and Chat Template
# 
# Configure the system prompt and chat template for the royal advisor role

# %% System Prompt
system_prompt = (
    "You are a royal advisor. Choose the option (1 or 2) that will best balance the kingdom's metrics "
    "(Piety, Stability, Power, Wealth) closest to 50. Think in less than 500 words. "
    "Respond with ONLY the option number - either 1 or 2."
)

# Using the model's default chat template
print("Using model's default chat template")
print(f"Template: {tokenizer.chat_template}")

# %% [markdown]
# ## Card Processing Functions
# 
# Functions for loading and processing card data

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
    prompt += f"1. {card['Left Choice']} {option1_effects}\n"
    prompt += f"2. {card['Right Choice']} {option2_effects}"
    
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
# ## Load Card Data
# 
# Load the card data from CSV or create sample data if not available

# %% Load Card Data
csv_path = "cards/cards.csv"  # Updated to use standard cards file
print(f"Loading card data from {csv_path}...")
try:
    cards_data = load_cards_data(csv_path)
except Exception as e:
    print(f"Error loading CSV: {e}")
    # Create a simple sample dataset if file not found
    data = {
        'Character': ['Royal Advisor', 'General', 'Priest'],
        'Prompt': ['We need more gold', 'We need more soldiers', 'The church needs funding'],
        'Left_Choice': ['Tax the people', 'Recruit farmers', 'Donate treasury gold'],
        'Right_Choice': ['Cut military spending', 'Hire mercenaries', 'Ignore the request'],
        'Left_Piety': [0, -10, 20],
        'Left_Stability': [-10, -10, 0],
        'Left_Power': [0, 10, -5],
        'Left_Wealth': [20, -5, -20],
        'Right_Piety': [-5, 0, -20],
        'Right_Stability': [0, 0, 10],
        'Right_Power': [-10, 20, 0],
        'Right_Wealth': [10, -20, 0],
    }
    cards_data = pd.DataFrame(data)

print(f"Loaded {len(cards_data)} cards")

# %% [markdown]
# ## Dataset Creation
# 
# Create training examples for the GRPO training process

# %% Create Dataset Function
def create_dataset(cards_data, num_examples=500):
    """Create a training dataset with examples of decision scenarios"""
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
    # Format dataset as system/user prompts with answers, and include metrics/card
    dataset = []
    for example in examples:
        dataset.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["prompt"]},
            ],
            "answer": {
                "choice": example["solution"],
                "metrics": example["metrics"],
                "card": example["card"],
            },
        })
    return Dataset.from_list(dataset)

# %% Generate Dataset
num_examples = 1000  # Adjust based on your needs
print(f"Creating dataset with {num_examples} examples...")
dataset = create_dataset(cards_data, num_examples)
print(f"Dataset size: {len(dataset)}")
print("Sample prompt:")
print(dataset[0]["prompt"][1]["content"])
print(f"Correct answer: {dataset[0]['answer']['choice']}")

# %% [markdown]
# ## Reward Functions
# 
# Define reward functions for the GRPO training process

# %% Main Reward Function
def reward_function_balanced(completions, **kwargs):
    """Reward is negative sum of distances from 50,50,50,50 after applying model's choice."""
    answer = kwargs.get('answer')
    logger.debug(f"--- reward_function START ---")
    logger.debug(f"Batch size: {len(completions)}")
    rewards = []
    for i, (completion_item, reference) in enumerate(zip(completions, answer)):
        # Extract the model's choice (should be just "1" or "2")
        if not completion_item or not isinstance(completion_item, list) or len(completion_item) == 0 or "content" not in completion_item[0]:
            logger.warning(f"  Unexpected completion format for item {i}: {completion_item}. Assigning raw_model_output as empty string.")
            raw_model_output = ""
        else:
            raw_model_output = completion_item[0]["content"].strip()
        # Extraction: get the first '1' or '2' after </think> (no fallback)
        if '<think>' in raw_model_output and '</think>' not in raw_model_output:
            logger.warning(f"  <think> tag not closed in raw_model_output: '{raw_model_output}'. Assigning strong penalty.")
            rewards.append(-200.0)
            continue
        choice = None
        # Find </think> and extract the first '1' or '2' after it
        after_think_match = re.search(r"</think>(.*)", raw_model_output, re.IGNORECASE | re.DOTALL)
        if after_think_match:
            after_text = after_think_match.group(1)
            m = re.search(r"[12]", after_text)
            if m:
                choice = m.group()
        if not choice:
            logger.warning(f"  No valid choice found after </think> in raw_model_output: '{raw_model_output}'. Assigning strong penalty.")
            rewards.append(-200.0)
            continue
        model_choice = choice
        # Log the prompt, raw output, and official answer for debugging
        try:
            # Try to get the prompt from reference if available
            prompt = reference.get("prompt") if isinstance(reference, dict) else None
            if prompt is None and isinstance(reference, dict):
                # Try to reconstruct prompt from metrics and card if possible
                card = reference.get("card")
                metrics = reference.get("metrics")
                if card is not None and metrics is not None:
                    prompt = format_card_as_prompt(card, metrics)
            logger.debug(f"[DEBUG][Item {i}] Prompt: {prompt}")
        except Exception as e:
            logger.warning(f"[DEBUG][Item {i}] Could not extract prompt: {e}")
            prompt = None
        logger.debug(f"[DEBUG][Item {i}] Raw model output: '{raw_model_output}'")
        logger.debug(f"[DEBUG][Item {i}] Extracted model choice: '{model_choice}'")
        try:
            official_answer = reference["choice"] if isinstance(reference, dict) and "choice" in reference else None
        except Exception as e:
            logger.warning(f"[DEBUG][Item {i}] Could not extract official answer: {e}")
            official_answer = None
        logger.debug(f"[DEBUG][Item {i}] Official answer: '{official_answer}'")
        try:
            metrics = reference["metrics"]
            card = reference["card"]
            left_effects = {
                'Piety': card['Left_Piety'],
                'Stability': card['Left_Stability'],
                'Power': card['Left_Power'],
                'Wealth': card['Left_Wealth'],
            }
            right_effects = {
                'Piety': card['Right_Piety'],
                'Stability': card['Right_Stability'],
                'Power': card['Right_Power'],
                'Wealth': card['Right_Wealth'],
            }
        except Exception as e:
            logger.warning(f"  Failed to access metrics/card in reference: {e} | reference: {reference}")
            rewards.append(-100.0)
            continue
        # Apply the model's choice
        if model_choice == "1":
            new_metrics = {k: metrics[k] + left_effects[k] for k in metrics}
        elif model_choice == "2":
            new_metrics = {k: metrics[k] + right_effects[k] for k in metrics}
        else:
            logger.warning(f"  No valid choice ('1' or '2') found in raw_model_output: '{raw_model_output}'. Assigning strong penalty.")
            rewards.append(-100.0)
            continue
        # Compute negative sum of distances from 50
        distance = sum(abs(new_metrics[k] - 50) for k in new_metrics)
        reward = -distance
        rewards.append(reward)
    logger.debug(f"Generated rewards for batch: {rewards}")
    logger.debug(f"--- reward_function END ---")
    return rewards

# %% Secondary Reward Functions
def match_correct_format(completions, **kwargs):
    """Reward model for giving just a single number as response"""
    scores = []
    for completion in completions:
        response = completion[0]["content"].strip()
        
        # Check if response contains only "1" or "2" (or both)
        if re.match(r"^[12]$", response):
            scores.append(1.0)  # Perfect format
        elif "1" in response or "2" in response:
            scores.append(0.5)  # Contains a correct option but with extra text
        else:
            scores.append(-1.0)  # No valid option
    
    return scores

def penalize_lengthy_responses(completions, **kwargs):
    """Penalize verbose responses - we want just 1 or 2, and never over 500 words"""
    scores = []
    for completion in completions:
        response = completion[0]["content"].strip()
        word_count = len(response.split())

        # Strong penalty for going over 500 words
        if word_count > 500:
            scores.append(-1.0)  # Very strong penalty
        else:
            scores.append(0.0)  #  Within the limit
    return scores

# %% [markdown]
# ## Calculate Max Prompt Length
# 
# Calculate the maximum prompt length for GRPO configuration

# %% Calculate Max Prompt Length
print("Calculating maximum prompt length...")
tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

maximum_length = int(np.quantile(tokenized["L"], 0.9))
print(f"Max prompt length: {maximum_length}")

# %% [markdown]
# ## GRPO Training Configuration
# 
# Set up the GRPO trainer with appropriate parameters

# %% GRPO Configuration
from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p=0.0,
    top_p=0.95,
    top_k=20,
    seed=3407,
    temperature=0.6,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=0.7,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=4,  # Generate 4 different completions per prompt
    max_prompt_length=4096,  # Increased by 4x for much longer prompts
    max_completion_length=2048,  # Increased by 4x for much longer outputs
    num_train_epochs=1,  # Number of training epochs
    output_dir="dynast_ai_grpo_model",
    save_steps=100,
    report_to=["wandb"],
    # Add reward weights - 80% for correctness, 20% for brevity
    reward_weights=[0.8, 0.2],
)

# %% [markdown]
# ## Train the Model
# 
# Create and run the GRPO trainer

# %% Run GRPO Training
print("Starting GRPO training...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_function_balanced,  # Main reward function
        # match_correct_format,  # Reward for correct format
        penalize_lengthy_responses,  # Penalize verbose responses
    ],
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()

# %% [markdown]
# ## Save the Trained Model
# 
# Save the LoRA adapter weights

# %% Save Model
print("Saving model...")
model.save_lora("dynast_ai_grpo_lora")
print("Training completed and model saved!")

# %% [markdown]
# ## Upload LoRA weights to Hugging Face Hub
# 
# Upload the LoRA weights to Hugging Face Hub

# %% Upload LoRA weights to Hugging Face Hub
# Set your repo name and organization/user
repo_id = "Slyracoon23/dynastai-grpo-lora"  # TODO: Change to your Hugging Face username/org and desired repo name

# (Optional) Create the repo if it doesn't exist
try:
    create_repo(repo_id, private=False)  # Set private=True if you want a private repo
except Exception as e:
    print(f"Repo may already exist or error occurred: {e}")

# Upload the LoRA folder
upload_folder(
    repo_id=repo_id,
    folder_path="dynast_ai_grpo_lora",
    path_in_repo="",  # Upload at root of repo
    commit_message="Upload LoRA adapter weights for DynastAI GRPO"
)
print(f"LoRA weights uploaded to Hugging Face Hub at https://huggingface.co/{repo_id}")

# %% [markdown]
# ## Testing the Model
# 
# Generate responses using the trained model

# %% Generate Test Example
def generate_test_example():
    """Generate a random test example"""
    card = cards_data.sample(1).iloc[0]
    
    # Generate random metrics between 10 and 90
    current_metrics = {
        'Piety': np.random.randint(10, 90),
        'Stability': np.random.randint(10, 90),
        'Power': np.random.randint(10, 90),
        'Wealth': np.random.randint(10, 90)
    }
    
    user_prompt = format_card_as_prompt(card, current_metrics)
    optimal_choice = determine_optimal_choice(card, current_metrics)
    
    return {
        "prompt": user_prompt,
        "optimal_choice": optimal_choice,
        "metrics": current_metrics,
        "card": card
    }

# %% Test Model
test_example = generate_test_example()
print(f"Test Example:\n{test_example['prompt']}")
print(f"Optimal choice: {test_example['optimal_choice']}")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": test_example['prompt']},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature=0.6,
    top_k=20,
    top_p=0.95,
    max_tokens=2048,  # Increased for much longer outputs
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

print("Generating response with trained model...")
output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("dynast_ai_grpo_lora"),
)[0].outputs[0].text

print(f"Model response: {output}")
print(f"Correct answer: {test_example['optimal_choice']}")

# %% [markdown]
# ## Evaluation Section
# 
# Evaluate the model on the entire dataset and print accuracy

# %% Evaluate Model on Entire Dataset
print("\nEvaluating model on the entire dataset...")
from tqdm import tqdm  # Add tqdm for progress bar
correct = 0
n = min(10, len(dataset))  # Limit to 10 examples

# Load LoRA weights once for all generations
lora_request = model.load_lora("dynast_ai_grpo_lora")

for idx, example in enumerate(tqdm(dataset[:n], desc="Evaluating prompts")):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["prompt"][1]["content"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=lora_request,  # Use preloaded LoRA
    )[0].outputs[0].text.strip()
    # Extract the first occurrence of "1" or "2"
    model_choice = None
    for char in output:
        if char in ["1", "2"]:
            model_choice = char
            break
    # Print the prompt, correct answer, and model output after every evaluation
    print(f"\nPrompt {idx+1}:\n{example['prompt'][1]['content']}")
    print(f"Correct answer: {example['answer']['choice']}")
    print(f"Model output: {output}")
    if model_choice == example['answer']['choice']:
        correct += 1

print(f"\nEvaluation complete. Accuracy: {correct}/{n} = {correct/n:.2%}")
# %%
