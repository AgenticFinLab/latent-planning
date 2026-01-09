"""
The evaluation to get the performance of the finetuned model on the different datasets.
"""

import os
import yaml
import argparse


import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_config):
    """Load custom Hugging Face model from checkpoint"""
    checkpoint_path = model_config["checkpoint_path"]

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_config["checkpoint_path"],
        device_map="auto",  # Automatically uses GPU if available
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    return model, tokenizer


def generate_text(model, tokenizer, data_loader, generation_config):
    """Run text generation using the model's native device"""
    model.eval()
    all_generated = []

    with torch.no_grad():
        for batch in data_loader:
            # Move inputs to same device as model
            inputs = batch["input_ids"].to(model.device)

            outputs = model.generate(
                inputs,
                temperature=generation_config.get("temperature", 1.0),
                top_k=generation_config.get("top_k", 50),
                max_new_tokens=generation_config.get("max_new_tokens", 100),
                do_sample=generation_config.get("do_sample", True),
                pad_token_id=tokenizer.pad_token_id,
            )

            # Move outputs back to CPU for decoding
            decoded = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
            all_generated.extend(decoded)

    return all_generated


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config.yml path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-s", "--save", required=True, help="Save path")
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)
    model, tokenizer = load_model(config["model"])

    # Load the dataset
    ## Stage 3. Load and process the data
    dataset = load_dataset(path=args.dataset)

    dataset = dataset.map(
        lambda x: reason_generator.create_plan_reason_func(x, concept_learner),
        batched=True,
    )


if __name__ == "__main__":
    main()
