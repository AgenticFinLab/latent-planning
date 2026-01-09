"""
Transform the dataset to support the model fine-tuning on the plan generation.

The original data loaded from the huggingface mainly has the sample that contains:
'question', 'Steps', 'Plans'.
'Steps' and 'Plans' have the same structure, i.e. the list containing the step-wise step and the corresponding plans.

Thus, each sample is transformed to many plan samples while each plan sample
is prepared for the training and thus mainly contains:

Question, previous steps, the next-step plan.
"""

import os
import json
import logging


from datasets import load_dataset

from projinit.config import Config
from projinit import platform_init

from modules.pGen_data import organize_plan_samples, organize_plan_reason_samples


def _main():
    """Main session to finetune the model."""
    # Set the platforms
    platform_init.InitializePlatforms().login_accounts()

    ## Stage 1. Get the learning configuration
    train_config = Config.items_to_dict(Config().train._asdict())
    data_synthetic_config = train_config["data_synthetic"]
    synthesized_path = data_synthetic_config["synthesized_path"]
    synthesized_name = data_synthetic_config["synthesized_name"]
    train_filename = data_synthetic_config["ft_train_dataname"]
    test_filename = data_synthetic_config["ft_test_dataname"]
    train_e_filename = data_synthetic_config["ft_train_e_dataname"]
    test_e_filename = data_synthetic_config["ft_test_e_dataname"]

    save_path = f"{synthesized_path}/{synthesized_name}"
    os.makedirs(save_path, exist_ok=True)
    train_file_path = f"{save_path}/{train_filename}"
    test_file_path = f"{save_path}/{test_filename}"
    train_e_file_path = f"{save_path}/{train_e_filename}"
    test_e_file_path = f"{save_path}/{test_e_filename}"

    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    hf_data = data_synthetic_config["hf_dataname"]
    dataset = load_dataset(hf_data, token=HUGGINGFACE_TOKEN)

    if not os.path.exists(train_file_path):
        logging.info("Preparing data from %s", train_file_path)
        train_dataset = dataset["train"]
        train_plan_samples = organize_plan_samples(train_dataset)
        with open(train_file_path, "w", encoding="utf-8") as json_file:
            json.dump(train_plan_samples, json_file)
        logging.info(
            "Prepared finetune trainset with %d samples to %s\n",
            len(train_plan_samples),
            train_file_path,
        )

    if not os.path.exists(test_file_path):
        logging.info("Preparing data from %s", test_file_path)
        test_dataset = dataset["test"]
        test_plan_samples = organize_plan_samples(test_dataset)
        with open(test_file_path, "w", encoding="utf-8") as json_file:
            json.dump(test_plan_samples, json_file)

        logging.info(
            "Prepared finetune testset with %d samples to %s\n",
            len(test_plan_samples),
            test_file_path,
        )

    if not os.path.exists(train_e_file_path):
        logging.info("Preparing data from %s", train_e_file_path)
        train_dataset = dataset["train"]
        train_plan_samples = organize_plan_samples(train_dataset, plan_name="EPlans")
        with open(train_e_file_path, "w", encoding="utf-8") as json_file:
            json.dump(train_plan_samples, json_file)
        logging.info(
            "Prepared finetune trainset with %d samples to %s\n",
            len(train_plan_samples),
            train_e_file_path,
        )

    if not os.path.exists(test_e_file_path):
        logging.info("Preparing data from %s", test_e_file_path)
        test_dataset = dataset["test"]
        test_plan_samples = organize_plan_samples(test_dataset, plan_name="EPlans")
        with open(test_e_file_path, "w", encoding="utf-8") as json_file:
            json.dump(test_plan_samples, json_file)

        logging.info(
            "Prepared finetune testset with %d samples to %s\n\n",
            len(test_plan_samples),
            test_e_file_path,
        )

    ### Prepare the plan-based reasoning samples
    train_filename = data_synthetic_config["ft_train_plan_reason_dataname"]
    test_filename = data_synthetic_config["ft_test_plan_reason_dataname"]
    train_e_filename = data_synthetic_config["ft_train_e_plan_reason_dataname"]
    test_e_filename = data_synthetic_config["ft_test_e_plan_reason_dataname"]
    train_file_path = f"{save_path}/{train_filename}"
    test_file_path = f"{save_path}/{test_filename}"
    train_e_file_path = f"{save_path}/{train_e_filename}"
    test_e_file_path = f"{save_path}/{test_e_filename}"

    if not os.path.exists(train_file_path):
        logging.info("Preparing data from %s", train_file_path)
        train_dataset = dataset["train"]
        train_plan_samples = organize_plan_reason_samples(train_dataset)
        with open(train_file_path, "w", encoding="utf-8") as json_file:
            json.dump(train_plan_samples, json_file)
        logging.info(
            "Prepared finetune trainset with %d samples to %s\n",
            len(train_plan_samples),
            train_file_path,
        )

    if not os.path.exists(test_file_path):
        logging.info("Preparing data from %s", test_file_path)
        test_dataset = dataset["test"]
        test_plan_samples = organize_plan_reason_samples(test_dataset)
        with open(test_file_path, "w", encoding="utf-8") as json_file:
            json.dump(test_plan_samples, json_file)

        logging.info(
            "Prepared finetune testset with %d samples to %s\n",
            len(test_plan_samples),
            test_file_path,
        )

    if not os.path.exists(train_e_file_path):
        logging.info("Preparing data from %s", train_e_file_path)
        train_dataset = dataset["train"]
        train_plan_samples = organize_plan_reason_samples(
            train_dataset, plan_name="EPlans"
        )
        with open(train_e_file_path, "w", encoding="utf-8") as json_file:
            json.dump(train_plan_samples, json_file)
        logging.info(
            "Prepared finetune trainset with %d samples to %s\n",
            len(train_plan_samples),
            train_e_file_path,
        )

    if not os.path.exists(test_e_file_path):
        logging.info("Preparing data from %s", test_e_file_path)
        test_dataset = dataset["test"]
        test_plan_samples = organize_plan_reason_samples(
            test_dataset, plan_name="EPlans"
        )
        with open(test_e_file_path, "w", encoding="utf-8") as json_file:
            json.dump(test_plan_samples, json_file)

        logging.info(
            "Prepared finetune testset with %d samples to %s\n",
            len(test_plan_samples),
            test_e_file_path,
        )


if __name__ == "__main__":

    _main()
