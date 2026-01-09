"""
The finetuning process to generate the latent plans created by the
trained encoder + quantizer.

The objective of the fine-tuning is the next-token prediction, next
latent plan prediction.

Training:
1. Create a new dataset by using
groundtruth plans --> encoder -> quantizer --> indices of latent plans

2. Finetune a decoder-only LLM
question   -->
steps      -->  LLM   --> 1-shift indices
indices    -->
indicates  -->
"""

import random
import wandb
import torch
from datasets import load_dataset

from trl import SFTTrainer, SFTConfig

from transformers import DataCollatorForSeq2Seq

from modules import pGen_data
from modules import pGen_reasoner
from modules.dataset_utils import train_on_responses_only
from projinit import config
from projinit.platform_init import InitializePlatforms
from trlm.util import template_tools

import callbacks
import init_project


def _main():
    """Main session to finetune the model."""
    # major_version, minor_version = torch.cuda.get_device_capability()

    #########################################################
    ## Define the project and set the platforms
    InitializePlatforms().login_accounts()
    # Create the project information
    proj = init_project.pGenProjectInfo(project_type="reasoner")
    wandb_run = proj.create_wandb(entity="LatentPlanReasoner")

    torch.cuda.empty_cache()

    ## Stage 2. Define the model
    reason_generator, concept_learner = pGen_reasoner.define_model(
        proj.model_config,
        train_config=proj.train_config,
        wandb_run=wandb_run,
        plan_status=proj.plan_status,
    )

    ## Stage 3. Load and process the data
    data_files = {
        "train": proj.train_dataname,
        "test": proj.test_dataname,
    }
    dataset = load_dataset(
        "json",
        data_dir=f"{proj.data_folder}/{proj.synthesized_data_name}",
        data_files=data_files,
    )

    # Note that the batched and the batch size of the map function does
    # not influence that of the training process!
    train_dataset = dataset["train"].map(
        lambda x: reason_generator.create_plan_reason_func(x, concept_learner),
        batched=True,
    )
    test_dataset = dataset["test"].map(
        lambda x: reason_generator.create_plan_reason_func(x, concept_learner),
        batched=True,
    )

    text_train_data = dataset["train"]
    text_test_data = dataset["test"]

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

    message = sample["plan_reason_message"]
    sample_roles, row_data = pGen_data.extract_message(message)
    sample_roles.append("text_input")
    row_data.append(sample["text"])
    # log the table to wandb
    wandb_run.log(
        {"training_samples": wandb.Table(columns=sample_roles, data=[row_data])}
    )
    ## Stage 4. Fine-tune the model with Huggingface TRL's SFTTrainer
    # Expected number of training steps
    batch_size = proj.train_config["per_device_train_batch_size"]
    gradient_size = proj.train_config["gradient_accumulation_steps"]
    n_gpus = (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else 1 if torch.backends.mps.is_available() else 0
    )
    eval_batch_size = proj.eval_config["per_device_eval_batch_size"]
    total_steps = num_train / (batch_size * gradient_size * n_gpus)
    n_eval_steps = 100 if total_steps > 1000 else int(total_steps / 10)

    # Remove the concept_learner to save the space
    if concept_learner is not None:
        del concept_learner
    torch.cuda.empty_cache()

    # Set up the training arguments
    train_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_size,
        label_names=["labels"],  # Used the prediction/evaluation
        dataset_text_field="text",
        max_seq_length=proj.model_config["reasoner"]["max_seq_length"],
        packing=False,  # Can make training 5x faster for short sequences.
        warmup_steps=5,
        num_train_epochs=proj.train_config["epoch"],
        learning_rate=proj.train_config["learning_rate"],
        # fp16=False if major_version >= 8 else True,
        # bf16=True if major_version >= 8 else False,
        logging_steps=proj.log_config["log_steps"],
        logging_first_step=True,
        optim="adamw_8bit",
        weight_decay=proj.train_config["weight_decay"],
        lr_scheduler_type=proj.train_config["lr_scheduler"],
        dataset_num_proc=2,
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

    tokenizer = reason_generator.tokenizer
    trainer = SFTTrainer(
        model=reason_generator,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=train_args,
    )

    # As different llms use different prompt templates, we need to set the
    # instruction and response accordingly to avoid the issues of
    # 'NAN loss' and 'ZeroDivisionError: division by zero'
    instruction, response = template_tools.get_template_parts(
        model_name=proj.model_config["reasoner"]["model_name"]
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

    # Instantiate the new logging callback, passing it the Trainer object
    selected_indices = range(0, 100)
    evals_callback = callbacks.ProgressDetailsCallback(
        trainer=trainer,
        val_subset=trainer.eval_dataset.select(selected_indices),
        train_subset=trainer.train_dataset.select(selected_indices),
        text_train_subset=text_train_data.select(selected_indices),
        text_val_subset=text_test_data.select(selected_indices),
        num_fixed=3,
        num_random=2,
        folder_path=proj.log_config["result_path"],
    )

    # Add the callback to the Trainer
    trainer.add_callback(evals_callback)

    # print(trainer.train_dataset[5]["labels"])
    trainer_stats = trainer.train()

    # Finish wandb run
    # The detailed run history is generated when we finish the Weights & Biases run.
    wandb_run.finish()
    config.Config.set_records(status="Completed")


if __name__ == "__main__":

    _main()
