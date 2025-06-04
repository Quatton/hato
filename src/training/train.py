import json
from datasets import Dataset
import os
from prompt.template import system_prompt
import PIL.Image


from vlmgrpo import VLMGRPOTrainer
from unsloth import FastVisionModel
from infer.qwen import pmodel, tokenizer
from training.reward import match_format_exactly, check_answer

# Load the JSON file
with open('datasets/tokyo-2000-address.json', 'r') as f:
    data = json.load(f)

# Extract the list of entries
entries = data.get("customCoordinates", [])

# Create a list of dictionaries for the dataset
dataset_data = []
for i, entry in enumerate(entries):
    # Construct the image path
    image_path = f"datasets/out/{i}_{entry['panoId']}.jpg"

    # Check if the image file exists
    if os.path.exists(image_path):
        dataset_data.append({
            "panoId": entry.get("panoId"),
            "address": entry.get("address"),
            "image_path": image_path # Add the image path
        })
    else:
        print(f"Warning: Image not found at {image_path} for entry index {iter}")

# Create a Hugging Face Dataset from the list of dictionaries
dataset = Dataset.from_list(dataset_data).train_test_split(0.2)

dataset_with_prompt = dataset.map(
    lambda x: {
        "prompt": [
                    {"role": "system", "content": [
                        {"type": "text", "text": system_prompt} ]},
                    {"role": "user", "content": [
                        {"type": "image"}
                    ]}
        ],
        "image": PIL.Image.open(x["image_path"]),
        "answer": {
            "ward": x["address"]["city"]["name"],
            "town": x["address"]["oaza"]["name"]
        }
    }
)

from trl import GRPOConfig
training_args = GRPOConfig(
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    # max_prompt_length = 2048,
    max_completion_length = 2048,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 10,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",

    # For optional training + evaluation
    fp16_full_eval = True,
    per_device_eval_batch_size = 4,
    eval_accumulation_steps = 1,
    eval_strategy = "steps",
    eval_steps = 1,
)

FastVisionModel.for_training(pmodel) # Enable for training!

trainer = VLMGRPOTrainer(
    model = pmodel,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        check_answer,
    ],
    args = training_args,
    # train_dataset = dataset_with_prompt["train"],
    reward_processing_classes = tokenizer, #Here also
    grad_verbose = True,
    # For optional training + evaluation
    train_dataset = dataset_with_prompt["train"],
    eval_dataset = dataset_with_prompt["test"],
)

trainer.train()