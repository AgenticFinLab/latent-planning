"""
This is to present the similarity between the plans and the steps
"""

import os
import json
import logging

import plot_embed
import result_tools
import compute_encoding
from commons import target_math_categories

import h5py
import numpy as np

from trlm.dataset import define_dataset

logging.basicConfig(level=logging.INFO)


# Set the path and filenames
project_path = "examples/LatentPlan/preExperiments"
saved_path = f"{project_path}/step_plan_sim"
step_sim_path = f"{project_path}/step_sim"
plan_sim_path = f"{project_path}/plan_sim"
selected_filename = "selected_samples.json"
decomposed_step_filepath = f"{project_path}/step_sim/decomposed_answers.json"
summarized_plan_filepath = f"{project_path}/plan_sim/plans.json"

selected_samples = {}

os.makedirs(saved_path, exist_ok=True)
file_path = f"{project_path}/{selected_filename}"

# Get the indexes of the selected samples and the decomposed steps
with open(file_path, "r", encoding="utf-8") as json_file:
    selected_samples = json.load(json_file)
with open(decomposed_step_filepath, "r", encoding="utf-8") as json_file:
    decomposed_steps = json.load(json_file)
with open(summarized_plan_filepath, "r", encoding="utf-8") as json_file:
    summarized_plans = json.load(json_file)

logging.info("Loaded the decomposed steps from %s", decomposed_step_filepath)
logging.info("Loaded the summarized plans from %s", summarized_plan_filepath)

selected_samples = {
    category: selected_samples[category] for category in target_math_categories
}
decomposed_steps = {
    category: decomposed_steps[category] for category in target_math_categories
}
summarized_plans = {
    category: summarized_plans[category] for category in target_math_categories
}

# Get all data from the MATH dataset -- training set.
dataset = define_dataset(data_config={"data_name": "MATH"})
trainset = dataset.get_train_set()


######### Stage 1: Get the encodings of the steps
category_step_encodings = {}
category_step_pure_encodings = {}
for category, category_steps in decomposed_steps.items():
    encoding_path = f"{step_sim_path}/{category}_step_encodings.h5"
    pure_encoding_path = f"{step_sim_path}/{category}_step_pure_encodings.h5"

    with h5py.File(encoding_path, "r") as plan_f1, h5py.File(
        pure_encoding_path, "r"
    ) as plan_f2:
        plan_encodings = [np.array(plan_f1[key]) for key in plan_f1.keys()]
        pure_plan_encodings = [np.array(plan_f1[key]) for key in plan_f2.keys()]

        logging.info(
            "Step Encodings of %s exist, Loaded from %s",
            category,
            encoding_path,
        )

    category_step_encodings[category] = plan_encodings
    category_step_pure_encodings[category] = pure_plan_encodings


######### Stage 2: Get the encodings of the plans
category_plan_encodings = {}
category_plan_pure_encodings = {}
for category, category_plans in summarized_plans.items():
    encoding_path = f"{plan_sim_path}/{category}_plan_encodings.h5"
    pure_encoding_path = f"{plan_sim_path}/{category}_plan_pure_encodings.h5"

    with h5py.File(encoding_path, "r") as plan_f1, h5py.File(
        pure_encoding_path, "r"
    ) as plan_f2:
        plan_encodings = [np.array(plan_f1[key]) for key in plan_f1.keys()]
        pure_plan_encodings = [np.array(plan_f1[key]) for key in plan_f2.keys()]

        logging.info(
            "Plan Encodings of %s exist, Loaded from %s",
            category,
            encoding_path,
        )

    category_plan_encodings[category] = plan_encodings
    category_plan_pure_encodings[category] = pure_plan_encodings


######### Stage 3: Plot the relation between the plans and steps
for category in list(category_step_pure_encodings.keys()):
    # A list of 2D arrays
    # Get the step and plan distance distribution
    step_distance_distribution = compute_encoding.get_step_distributions(
        category_step_pure_encodings, category_name=category
    )
    plan_distance_distribution = compute_encoding.get_step_distributions(
        category_plan_pure_encodings, category_name=category
    )
    plot_embed.plot_question_step_relation(
        step_distance_distribution,
        plan_distance_distribution,
        save_path=saved_path,
        filename=f"{category}_steps_plans_relation.png",
        plot_config={
            "xlabel": "Reasoning Step ID",
            "ylabel": "Reasoning Plan distances",
        },
    )
