"""
This is to compute the encodings of the reasoning plans to check whether
1). the reasoning plans of the similar questions appear somehow similarity
2). the reasoning plans of the different questions appear significant differences.
"""

import os
import json
import logging

import latent_extract
import plot_embed

import result_tools
import compute_encoding
from commons import target_math_categories, abbre_math_categories

import h5py
import numpy as np

from iclp.old.dataset import define_dataset
import trlm.util.llm_tools as llm_tools

logging.basicConfig(level=logging.INFO)


logging.info("*" * 50)
logging.info("*" * 50)
logging.info(
    """This repo is to 1) summarize step-wise plans, and 2) compute their encodings on both the initial plans and ones without stopwords, i.e., pure, thus presenting inter- and intra- relations of them."""
)
logging.info("*" * 50)
logging.info("*" * 50)

# Set the path and filenames
project_path = "examples/LatentPlan/preExperiments"
saved_path = f"{project_path}/plan_sim"
selected_filename = "selected_samples.json"
decomposed_step_filepath = f"{project_path}/step_sim/decomposed_answers.json"
plan_filepath = f"{saved_path}/plans.json"
selected_samples = {}

os.makedirs(saved_path, exist_ok=True)
file_path = f"{project_path}/{selected_filename}"

# Get the indexes of the selected samples and the decomposed steps
with open(file_path, "r", encoding="utf-8") as json_file:
    selected_samples = json.load(json_file)
with open(decomposed_step_filepath, "r", encoding="utf-8") as json_file:
    decomposed_steps = json.load(json_file)
logging.info("Loaded the decomposed steps from %s", decomposed_step_filepath)

selected_samples = {
    category: selected_samples[category] for category in target_math_categories
}

decomposed_steps = {
    category: decomposed_steps[category] for category in target_math_categories
}


# Get all data from the MATH dataset -- training set.
dataset = define_dataset(data_config={"data_name": "MATH"})
trainset = dataset.get_train_set()

# Extract the features of them.
## 1. Get the answer of questions of different categories
category_questions = {}
for category_name, sample_indexes in selected_samples.items():
    questions = [trainset[idx]["question"] for idx in sample_indexes]
    category_questions[category_name] = questions


######### Stage 1: Summarize the plans of the decomposed steps.
summarized_plans = {}

if os.path.exists(plan_filepath):
    with open(plan_filepath, "r", encoding="utf-8") as json_file:
        summarized_plans = json.load(json_file)
        logging.info("Summarized plans exist, loaded from %s", plan_filepath)
else:
    for category, questions in category_questions.items():
        category_steps = decomposed_steps[category]
        summarized_plans[category] = []
        plans = []
        for q, steps in zip(questions, category_steps):
            # Summarize plans from the decomposed steps
            plans = llm_tools.llm_plan_summarization(
                question=q, steps=steps, model="gpt-4o"
            )

            summarized_plans[category].append(plans)
            assert len(steps) == len(plans)

        logging.info("Summarized plans for %s", category)

    with open(plan_filepath, "w", encoding="utf-8") as json_file:
        json.dump(summarized_plans, json_file)


######### Stage 2: Get the encodings of the plans
category_encodings = {}
category_pure_encodings = {}
for category, category_plans in summarized_plans.items():
    plan_encoding_path = f"{saved_path}/{category}_plan_encodings.h5"
    plan_pure_encoding_path = f"{saved_path}/{category}_plan_pure_encodings.h5"
    plan_encodings = []
    pure_plan_encodings = []
    if not os.path.exists(plan_encoding_path):

        for q_idx, q_plans in enumerate(category_plans):
            # Get the encodings of the plans
            features = latent_extract.sentence_encoding(input_sentences=q_plans)
            pure_features = latent_extract.sentence_encoding(
                input_sentences=latent_extract.filter_stopwords(q_plans)
            )
            plan_encodings.append(features)
            pure_plan_encodings.append(pure_features)

        logging.info("Obtained #%d encodings for %s", len(plan_encodings), category)
    else:
        with h5py.File(plan_encoding_path, "r") as plan_f1, h5py.File(
            plan_pure_encoding_path, "r"
        ) as plan_f2:
            plan_encodings = [np.array(plan_f1[key]) for key in plan_f1.keys()]
            pure_plan_encodings = [np.array(plan_f1[key]) for key in plan_f2.keys()]

            logging.info(
                "Plan Encodings of %s exist, Loaded from %s",
                category,
                plan_encoding_path,
            )

    category_encodings[category] = plan_encodings
    category_pure_encodings[category] = pure_plan_encodings

    # Save the encodings if not exist
    if not os.path.exists(plan_encoding_path):
        with h5py.File(plan_encoding_path, "w") as plan_f1, h5py.File(
            plan_pure_encoding_path, "w"
        ) as plan_f2:
            for q_idx, enc in enumerate(plan_encodings):
                plan_f1.create_dataset(f"q_{q_idx}", data=enc)
                plan_f2.create_dataset(f"q_{q_idx}", data=pure_plan_encodings[q_idx])


######### Stage 3: Plot the encodings
### Plot the encodings in the 2D spaces
plot_flag = False
plan_idx = 1
while plot_flag:
    # Obtain the encodings of the current step
    cur_plan_encodings, cur_plan_categories, _ = result_tools.extract_category_step(
        category_encodings,
        step_idx=plan_idx,
    )
    cur_plan_pure_encodings, cur_plan_categories, _ = (
        result_tools.extract_category_step(
            category_pure_encodings,
            step_idx=plan_idx,
        )
    )

    if len(cur_plan_encodings) <= 1:
        plot_flag = False
        continue

    # Plot them out
    plot_embed.plot_2d(
        encodings=cur_plan_encodings,
        encoding_labels=cur_plan_categories,
        label_legend="Category",
        save_path=saved_path,
        filename=f"category_plan_{plan_idx}_encodings_2d.png",
    )

    plot_embed.plot_2d(
        encodings=cur_plan_pure_encodings,
        encoding_labels=cur_plan_categories,
        label_legend="Category",
        save_path=saved_path,
        filename=f"category_plan_{plan_idx}_pure_encodings_2d.png",
    )
    logging.info(
        "Plotted #%d encodings in 2D space for the plan %d",
        len(cur_plan_encodings),
        plan_idx,
    )
    plan_idx += 1


######### Stage 3: Plot the relation between the question and plans
for category in list(category_pure_encodings.keys()):
    # A list of 2D arrays
    # Get the plan distance distribution
    # A Dict whose key is the plan index while the value is dict
    # containing, distribution and questions
    plan_distance_distribution = compute_encoding.get_step_distributions(
        category_pure_encodings, category_name=category
    )
    # Get the question distance distribution
    # 1. Get the question encoding
    q_encodings = latent_extract.get_encodings(
        sentences=None,
        encode_method=None,
        save_path=f"{project_path}/q_sim",
        encoding_name=f"{category}_question_pure_encoding",
    )
    # Replace the question indexes in step_distance_distribution with
    # their corresponding encodings
    plan_question_distribution = dict()
    for plan_idx, value in plan_distance_distribution.items():
        # Get the indexes of question has the current step
        _, _, cur_plan_questions = result_tools.extract_category_step(
            category_data=category_encodings, step_idx=plan_idx, category_name=category
        )
        target_q_encodings = q_encodings[cur_plan_questions]
        plan_question_distribution[plan_idx] = compute_encoding.compute_similarity(
            target_q_encodings
        )
    plot_embed.plot_question_step_relation(
        plan_distance_distribution,
        plan_question_distribution,
        save_path=saved_path,
        filename=f"{category}_plans_question_relation.png",
        plot_config={"xlabel": "Reasoning Plan ID", "ylabel": "Question Distance"},
    )


######### Stage 4: Plot the relation comparison between the question and plans
### Plot the encodings in the 2D spaces
plot_flag = True
plan_idx = 1
while plot_flag:
    # Obtain the encodings of the current step
    cur_plan_pure_encodings, cur_plan_categories, _ = (
        result_tools.extract_category_step(
            category_pure_encodings,
            step_idx=plan_idx,
        )
    )

    if len(cur_plan_pure_encodings) == 0:
        plot_flag = False
        continue
    distances = compute_encoding.compute_pairwise_similarity(
        encodings=cur_plan_pure_encodings
    )
    plot_embed.plot_distance_heatmap(
        distances=distances,
        labels=[abbre_math_categories[cate] for cate in cur_plan_categories],
        save_path=saved_path,
        filename=f"Plan_{plan_idx}_pure_encodings_relation",
        heatmap_config={
            "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
            "vmin": 0.4,
            "vmax": 0.8,
        },
    )

    # Get the question encodings of this step

    plan_question_encodings = []
    for category, category_content in category_pure_encodings.items():
        # Get the indexes of question has the current step
        _, _, cur_plan_questions = result_tools.extract_category_step(
            category_data=category_encodings, step_idx=plan_idx, category_name=category
        )
        q_encodings = latent_extract.get_encodings(
            sentences=None,
            encode_method=None,
            save_path=f"{project_path}/q_sim",
            encoding_name=f"{category}_question_pure_encoding",
        )
        plan_question_encodings.append(q_encodings[cur_plan_questions])

    plan_question_encodings = np.vstack(plan_question_encodings)
    plan_q_distances = compute_encoding.compute_pairwise_similarity(
        encodings=plan_question_encodings
    )

    plot_embed.plot_distance_heatmap(
        distances=plan_q_distances,
        labels=[abbre_math_categories[cate] for cate in cur_plan_categories],
        save_path=saved_path,
        filename=f"questions_of_plan_{plan_idx}_pure_encodings_relation",
        heatmap_config={
            "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
            "vmin": 0.4,
            "vmax": 0.8,
        },
    )

    plan_idx += 1


######### Stage 4: Compute the relation between the question and plan
# metric_results = compute_encoding.computer_clustering_metrics(
#     category_step_data=category_pure_encodings
# )

# print(metric_results)
