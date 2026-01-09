"""
The finetuning process to build the plan space based on the quantized autoencoder.

The objective of the fine-tuning is the plan reconstruction and the quantization loss.

The structure should be:

1. Create the dataset by making the user's input to be
(ignore chat template for simplify):
[plan placeholders] + [reconstruction indicates]

2. Training

placeholders + indicates + plans  --> decoder's tokenizer --> ids
--> re-get the strings --> encoder's tokenizer --> encoder
--> prev_quant_linear --> quantizer --> z_q --> extract the plans's
z_q

ids --> decoder's embeddings --> replace the placeholders with
plans's z_q --> decoder --> 1-shift next-token prediction.

"""

import random

import torch
from datasets import load_dataset

# from trl import SFTTrainer
from modules.dataset_utils import train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from modules import pGen_data
from modules import pGen_learner

from iclp.old.util import template_tools
from iclp.old.util.tools import ConfigLoader
import yaml
import argparse

import wandb


def _main():
    """Main session to finetune the model."""

    ## Stage 1. Define the project
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        default="",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=ConfigLoader) or {}

    wandb_run = wandb.init(
        entity="LatentPlanReasoner", project="LatentPlan", config=cfg
    )

    ## Stage 2. Define the model
    pgen_concept_learner = pGen_learner.define_model(
        cfg.get("model", {}), cfg.get("train", {}), wandb_run=wandb_run
    )

    ## Stage 3. Load and process the data
    train_config = cfg.get("train", {})
    data_synthetic_config = train_config.get("data_synthetic", {})
    data_folder = data_synthetic_config.get("synthesized_path", "")

    # Determine dataset names based on plan_type (logic adapted from init_project.py)
    plan_type = train_config.get("plan_type", "skeleton")
    if plan_type == "skeleton":
        trainset_name = data_synthetic_config.get("ft_train_dataname", "")
        testset_name = data_synthetic_config.get("ft_test_dataname", "")
    elif plan_type == "explicit":
        trainset_name = data_synthetic_config.get("ft_train_e_dataname", "")
        testset_name = data_synthetic_config.get("ft_test_e_dataname", "")
    else:
        raise ValueError(f"Unknown plan_type: {plan_type}")

    data_files = {
        "train": trainset_name,
        "test": testset_name,
    }
    dataset = load_dataset(
        "json",
        data_dir=data_folder,
        data_files=data_files,
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Note that the batched and the batch size of the map function does
    # not influence that of the training process!
    train_dataset = train_dataset.map(
        pgen_concept_learner.create_concept_func,
        batched=True,
    )
    test_dataset = test_dataset.map(
        pgen_concept_learner.create_concept_func,
        batched=True,
    )

    num_train = len(train_dataset)
    num_test = len(test_dataset)
    finetune_test_size = int(num_test / 10)
    text_table = wandb.Table(
        columns=["train_size", "test_size", "finetune_test_size"],
        data=[[num_train, num_test, finetune_test_size]],
    )
    wandb_run.log({"data_info": text_table})

    # We have randomly select samples for test
    some_indices = random.sample(range(0, num_test), finetune_test_size)
    test_dataset = test_dataset.select(some_indices)

    # Create an wandb Table to version the training predictions logged
    # This is to present the text input of the decoder part
    sample = train_dataset[20]
    message = sample["concept_message"]
    sample_roles, row_data = pGen_data.extract_message(message)
    sample_roles.append("input")
    row_data.append(sample["text"])
    sample_roles.append("placeholders")
    row_data.append(sample["placeholder"])
    # log the table to wandb
    wandb_run.log(
        {"training_samples": wandb.Table(columns=sample_roles, data=[row_data])}
    )

    ## Stage 4. Fine-tune the model with Huggingface TRL's SFTTrainer
    # Expected number of training steps
    batch_size = proj.train_config["per_device_train_batch_size"]
    gradient_size = proj.train_config["gradient_accumulation_steps"]
    n_gpus = torch.cuda.device_count()
    eval_batch_size = proj.eval_config["per_device_eval_batch_size"]
    total_steps = num_train / (batch_size * gradient_size * n_gpus)
    n_eval_steps = 100 if total_steps > 1000 else int(total_steps / 10)

    # Set up the training arguments
    train_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_size,
        label_names=["labels"],  # Used the prediction/evaluation
        warmup_steps=5,
        num_train_epochs=proj.train_config["epoch"],
        learning_rate=proj.train_config["learning_rate"],
        fp16=not proj.is_support_float16,
        bf16=proj.is_support_float16,
        logging_steps=proj.log_config["log_steps"],
        logging_first_step=True,
        optim="adamw_8bit",
        weight_decay=proj.train_config["weight_decay"],
        lr_scheduler_type=proj.train_config["lr_scheduler"],
        seed=3407,
        eval_strategy="steps",
        do_eval=True,
        eval_steps=n_eval_steps,
        eval_on_start=True,
        per_device_eval_batch_size=eval_batch_size,
        output_dir=proj.log_config["checkpoint_path"],
        report_to="wandb",
        disable_tqdm=True,
    )
    # We use the encoder's tokenizer
    tokenizer = pgen_concept_learner.decoder.tokenizer
    trainer = pGen_learner.ConceptLearnerTrainer(
        model=pgen_concept_learner,
        tokenizer=tokenizer,
        concept_weight=proj.train_config["concept_weight"],
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        max_seq_length=proj.model_config["max_seq_length"],
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=train_args,
        # callbacks=[pGen.GlobalStepLossLogger()],
    )

    # use Unsloth's train_on_completions method to only train on the assistant outputs and ignore the loss on the user's inputs.
    # As different llms use different prompt templates, we need to set the
    # instruction and response accordingly to avoid the issues of
    # 'NAN loss' and 'ZeroDivisionError: division by zero'
    instruction, response = template_tools.get_template_parts(
        model_name=proj.model_config["model_name"]
    )
    # This function will add the "labels" to the dataset!
    # Note: this is very important as we use 'DataCollatorForSeq2Seq'
    # above as the data collator, making a pre-defined "labels" should
    # be include as 'DataCollatorForSeq2Seq' will not create labels!!!
    # Thus, train_on_responses_only will be extremely important as "labels"
    # will be create within this function!!!
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction,
        response_part=response,
    )

    # print(trainer.train_dataset[5]["labels"])
    trainer_stats = trainer.train()

    # Finish wandb run
    # The detailed run history is generated when we finish the Weights & Biases run.
    wandb_run.finish()


if __name__ == "__main__":

    _main()
