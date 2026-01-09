"""
A running pipeline for the plan optimization phase of p-RAG.
"""

import os
import sys
import json
import logging
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass

from synthetic import llm_synthetic

from projinit.config import Config
from torch.utils.data import DataLoader
from transformers.utils import ModelOutput as FieldFrozenContainer

from trlm.dataset import registry as dataset_registry
from trlm.dataset.base import TextSample
from trlm.model import llms


@dataclass
class PlanReasonSample(FieldFrozenContainer):
    """A template of the samples to be shared in Huggingface."""

    # Unique ID of this sample
    main_id: str
    # The detailed information of the sample from the dataset
    sample_info: dict = None
    # Data Collection Timeline
    # "<Year>-<Month>-<Day>"
    timestamp: str = None
    # Text of the problem being solved
    question: str = None
    # The decomposed reasoning steps
    steps: List[List[str]] = None
    # The summarized reasoning plans
    skeleton_plans: List[List[str]] = None
    # The summarized reasoning explicit plans
    explicit_plans: List[List[str]] = None
    # groundtruth solution
    groundtruth: str = None


class PlanSyntheticPipeline:
    """
    A pipeline to accumulate the experiences for the SnowBall.
    """

    def __init__(self, phase: str = "train"):

        assert phase in ["train", "test"]
        self.phase = phase
        # The dataset to be processed
        self.dataset = None

        self.model_config = Config.items_to_dict(Config().model._asdict())
        self.data_config = Config.items_to_dict(Config().data._asdict())
        self.log_config = Config.items_to_dict(Config().logging._asdict())
        self.train_config = Config.items_to_dict(Config().train._asdict())

        # The loaded dataset
        self.loaded_dataset = None

        # The synthesized outputs
        self.synthesized_samples = []
        self.batch_idx = 1
        # We will save the samples to different blocks to
        # avoid the issue of out of memory
        # Each block will save the number of
        # batch_size * save_interval samples
        self.block_idx = 1

        # The samples whose decomposed answers do not match the
        # original answer
        self.unmatched_ids = []

        # Get the settings
        data_synthetic_config = self.train_config["data_synthetic"]
        synthesized_path = data_synthetic_config["synthesized_path"]
        synthesized_name = data_synthetic_config["synthesized_name"]

        hf_filename = (
            data_synthetic_config["hf_train_filename"]
            if self.phase == "train"
            else data_synthetic_config["hf_test_filename"]
        )
        self.save_path = f"{synthesized_path}/{synthesized_name}"
        os.makedirs(self.save_path, exist_ok=True)

        self.synthesized_data_path = f"{self.save_path}/{hf_filename}"

    def initialize(self):
        """Initialize the pipeline."""

        # Load the dataset
        self.loaded_dataset = dataset_registry.get(self.data_config, self.phase)

        logging.info(
            "---> Starting to synthesize samples in %s",
            self.train_config["data_synthetic"]["synthesized_name"],
        )
        # Reloaded the summarized explicit plans if exist
        if os.path.exists(self.synthesized_data_path):
            logging.info(
                "   - Exist synthesized samples in %s. Completed.",
                self.synthesized_data_path,
            )
            # exit the program
            sys.exit(0)

    def dump_batch_samples(
        self, batch_samples: List[TextSample], synthesized_outputs: Dict[str, list]
    ):
        """Dump the batch samples to the disk."""
        self.batch_idx += 1
        self.unmatched_ids.extend(
            [
                sample["main_id"]
                for idx, sample in enumerate(batch_samples)
                if not synthesized_outputs["matched"][idx]
            ]
        )

        self.synthesized_samples.extend(
            [
                PlanReasonSample(
                    main_id=item["main_id"],
                    sample_info=item["sample_info"],
                    timestamp=str(datetime.now().date()),
                    question=item["question"],
                    steps=steps,
                    skeleton_plans=skeleton_plans,
                    explicit_plans=explicit_plans,
                    groundtruth=str(item["groundtruth"]),
                )
                for item, steps, explicit_plans, skeleton_plans in zip(
                    batch_samples,
                    synthesized_outputs["steps"],
                    synthesized_outputs["explicit_plans"],
                    synthesized_outputs["skeleton_plans"],
                )
            ]
        )

        # Save the synthesized samples to the disk every several samples
        # to save the memory
        save_interval = self.train_config["data_synthetic"]["save_interval"]

        if self.batch_idx % save_interval == 0:
            path = f"{self.synthesized_data_path[:-5]}-{self.block_idx}.json"
            with open(path, "w", encoding="utf-8") as json_file:
                json.dump(self.synthesized_samples, json_file)

            self.synthesized_samples = []
            self.block_idx += 1

            # Save the unmatched ids to the disk
            unmatched_path = f"{self.save_path}/unmatched_ids.json"
            if len(self.unmatched_ids) > 0:
                with open(unmatched_path, "w", encoding="utf-8") as json_file:
                    json.dump(self.unmatched_ids, json_file)

    def synthesize(self):
        """Execute the pipeline to synthetic the plans."""
        logging.info(
            "   - Synthesizing samples from the loaded %s dataset.", self.phase
        )

        if "api" in self.model_config["model_type"].lower():
            logging.info("   - Using the API for inference.")
            synthesizer = llm_synthetic.DirectPlanSynthesize(
                decompose_step_config=self.model_config["decompose_step_config"],
                explicit_plan_config=self.model_config["explicit_plan_config"],
                skeleton_plan_config=self.model_config["skeleton_plan_config"],
            )
            dataloader = DataLoader(
                self.loaded_dataset,
                batch_size=self.train_config["data_synthetic"]["batch_size"],
                collate_fn=lambda batch: batch,
            )

            for batch in dataloader:
                # Inference the batch
                outputs = synthesizer(batch)
                self.dump_batch_samples(batch, outputs)

        else:
            # Note!
            #   Currently, the distributed mode has not been tested
            dis_inference = llms.RayDistributedInference(
                inference_operator=lambda: llm_synthetic.DirectPlanSynthesize(
                    decompose_step_config=self.model_config["decompose_step_config"],
                    explicit_plan_config=self.model_config["explicit_plan_config"],
                    skeleton_plan_config=self.model_config["skeleton_plan_config"],
                ),
                distributed_config={},
            )
            ds_results = dis_inference.inference(self.loaded_dataset)

            # Create the target sample
            all_outputs = ds_results.take_all()
            # Note! This part has not been tested
            for sample, outputs in zip(self.loaded_dataset, all_outputs):
                self.dump_batch_samples([sample], outputs)

        # Save the synthesized samples to the disk
        path = f"{self.synthesized_data_path[:-5]}-{self.block_idx}.json"
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(self.synthesized_samples, json_file)
            self.synthesized_samples = []
            self.batch_idx += 1

        # Save the unmatched ids to the disk
        unmatched_path = f"{self.save_path}/unmatched_ids.json"
        if len(self.unmatched_ids) > 0:
            with open(unmatched_path, "w", encoding="utf-8") as json_file:
                json.dump(self.unmatched_ids, json_file)
