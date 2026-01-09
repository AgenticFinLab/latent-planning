"""
A collection of callbacks used during the finetuning.

https://docs.wandb.ai/guides/integrations/huggingface/

# Add the callback to the trainer
trainer.add_callback(progress_callback)
"""

import os
import random

import pandas as pd
from transformers.trainer_callback import TrainerCallback

import inference_ops


class ProgressDetailsCallback(TrainerCallback):
    """
    Custom WandbCallback to log model predictions during evaluation.

    This callback logs model predictions and labels as CSV files saved locally.
    It logs predictions from both a subset of the validation dataset and,
    if provided, from a subset of the training dataset.
    Each CSV includes the current epoch and the global step.
    """

    def __init__(
        self,
        trainer,
        val_subset,
        train_subset,
        text_val_subset,
        text_train_subset,
        num_fixed: int = 3,
        num_random: int = 2,
        folder_path="results",
    ):
        """
        Initializes the ProgressDetailsCallback.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            train_dataset (Dataset, optional): The training dataset. If provided,
                predictions from training samples will also be logged.
            num_samples (int, optional): Number of samples to select from each dataset for prediction.
            folder_path (str, optional): Folder path where CSV results will be saved.
        """
        super().__init__()
        self.trainer = trainer
        self.val_subset = val_subset
        self.train_subset = train_subset

        self.n_val = len(self.val_subset)
        self.n_tr = len(self.train_subset)
        self.fixed_indices = random.sample(range(0, self.n_val), num_fixed)

        self.num_random = num_random
        self.folder_path = os.path.join(folder_path, "ProgressDetails")
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def on_evaluate(self, args, state, control, **kwargs):
        # Call the base class method to log standard metrics.
        super().on_evaluate(args, state, control, **kwargs)
        if state.is_world_process_zero:
            # ----- Log validation sample predictions -----
            val_indices = random.sample(range(0, self.n_val), self.num_random)
            tr_indices = random.sample(range(0, self.n_tr), self.num_random)

            val_indices = self.fixed_indices + val_indices
            tr_indices = self.fixed_indices + tr_indices

            val_subset = self.val_subset.select(val_indices)
            tr_subset = self.train_subset.select(tr_indices)

            val_outputs: inference_ops.EvalLoopInputOutputWith = (
                inference_ops.inference_with_trainer(
                    trainer=self.trainer,
                    dataset=val_subset,
                )
            )
            # Create the DataFrame directly from the NamedTuple's dictionary.
            val_df = pd.DataFrame(val_outputs.to_csv_dict())

            # Add the extra columns for every row
            val_df["epoch"] = state.epoch
            val_df["global_step"] = state.global_step

            # Save validation predictions to CSV in the specified folder.
            val_csv_path = os.path.join(self.folder_path, "sample_val_predictions.csv")
            if os.path.exists(val_csv_path):
                val_df.to_csv(val_csv_path, mode="a", index=False, header=False)
            else:
                val_df.to_csv(val_csv_path, mode="w", index=False, header=True)

            # ----- Log training sample predictions (if provided) -----
            tr_outputs: inference_ops.EvalLoopInputOutputWith = (
                inference_ops.inference_with_trainer(
                    trainer=self.trainer,
                    dataset=tr_subset,
                )
            )
            # Create the DataFrame directly from the NamedTuple's dictionary.
            tr_df = pd.DataFrame(tr_outputs.to_csv_dict())
            # Add the extra columns for every row
            tr_df["epoch"] = state.epoch
            tr_df["global_step"] = state.global_step

            # Save training predictions to CSV in the specified folder.
            train_csv_path = os.path.join(
                self.folder_path, "sample_train_predictions.csv"
            )
            if os.path.exists(train_csv_path):
                tr_df.to_csv(train_csv_path, mode="a", index=False, header=False)
            else:
                tr_df.to_csv(train_csv_path, mode="w", index=False, header=True)
