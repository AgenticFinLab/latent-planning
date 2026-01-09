"""
This is to compute the encodings of the reasoning steps to check whether
1). the reasoning steps of the similar questions appear somehow similarity
2). the reasoning steps of the different questions appear significant differences.
"""

import re
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

# Set the path and filenames
project_path = "examples/LatentPlan/preExperiments"
saved_path = f"{project_path}/step_sim"
selected_filename = "selected_samples.json"
decomposed_step_filepath = f"{saved_path}/decomposed_answers.json"
selected_samples = {}

os.makedirs(saved_path, exist_ok=True)
file_path = f"{project_path}/{selected_filename}"

with open(file_path, "r", encoding="utf-8") as json_file:
    selected_samples = json.load(json_file)

selected_samples = {
    category: selected_samples[category] for category in target_math_categories
}

# Get all data from the MATH dataset -- training set.
dataset = define_dataset(data_config={"data_name": "MATH"})
trainset = dataset.get_train_set()

# Extract the features of them.
## 1. Get the answer of questions of different categories
category_questions = {}
category_answers = {}
for category_name, sample_indexes in selected_samples.items():
    answers = [trainset[idx]["answer"] for idx in sample_indexes]
    questions = [trainset[idx]["question"] for idx in sample_indexes]
    category_questions[category_name] = questions
    category_answers[category_name] = answers


######### Stage 1: Decompose the answer into individual steps
decomposed_answers = {}

if os.path.exists(decomposed_step_filepath):
    with open(decomposed_step_filepath, "r", encoding="utf-8") as json_file:
        decomposed_answers = json.load(json_file)
    logging.info("Decomposed answers exist, loaded from %s", decomposed_step_filepath)
else:
    for category, questions in category_questions.items():
        answers = category_answers[category]
        decomposed_answers[category] = []
        for q, answer in zip(questions, answers):
            # Decompose the answer into multiple steps
            steps = llm_tools.llm_answer_decomposition(
                question=q, answer=answer, model="gpt-4o"
            )
            # Regex to capture each step and its content
            steps_list = re.findall(
                r"((?:Step|step)\s*\d+\s*:\s*.*?)(?=(Step\s*\d+\s*:|step\s*\d+\s*:|$))",
                steps,
                re.DOTALL,
            )

            # Extract only the step content (first part of each match)
            steps_list = [step[0].strip() for step in steps_list]
            decomposed_answers[category].append(steps_list)
            assert len(re.findall(r"Step \d+:", steps)) == len(steps_list)

        logging.info("Decomposed #%d answers for %s", len(answers), category)

    with open(decomposed_step_filepath, "w", encoding="utf-8") as json_file:
        json.dump(decomposed_answers, json_file)

    logging.info("Saved Decomposed answers to %s", decomposed_step_filepath)

decomposed_answers = {
    category: decomposed_answers[category] for category in target_math_categories
}

######### Stage 2: Get the encodings of the steps and the plans
# category_encodings: A dict whose key is the category name corresponding to
#   a list containing a series of 2D arrays while each 2D array containing the
#   encodings of steps of this question.
# category_pure_encodings: Same as the category_encodings.
category_encodings = {}
category_pure_encodings = {}
for category, category_steps in decomposed_answers.items():
    step_encoding_path = f"{saved_path}/{category}_step_encodings.h5"
    step_pure_encoding_path = f"{saved_path}/{category}_step_pure_encodings.h5"
    step_encodings = []
    pure_step_encodings = []
    if not os.path.exists(step_encoding_path):
        for q_idx, q_steps in enumerate(category_steps):
            # Get the encodings of the steps
            features = latent_extract.sentence_encoding(input_sentences=q_steps)
            pure_features = latent_extract.sentence_encoding(
                input_sentences=latent_extract.filter_stopwords(q_steps)
            )
            step_encodings.append(features)
            pure_step_encodings.append(pure_features)

        logging.info("Obtained #%d encodings for %s", len(step_encodings), category)

    else:
        with h5py.File(step_encoding_path, "r") as step_f1, h5py.File(
            step_pure_encoding_path, "r"
        ) as step_f2:
            step_encodings = [np.array(step_f1[key]) for key in step_f1.keys()]
            pure_step_encodings = [np.array(step_f1[key]) for key in step_f2.keys()]
        logging.info(
            "Step Encodings of %s exist, Loaded from %s", category, step_encoding_path
        )

    category_encodings[category] = step_encodings
    category_pure_encodings[category] = pure_step_encodings

    # Save the encodings if not exist
    if not os.path.exists(step_encoding_path):
        with h5py.File(step_encoding_path, "w") as step_f1, h5py.File(
            step_pure_encoding_path, "w"
        ) as step_f2:
            for q_idx, enc in enumerate(step_encodings):
                step_f1.create_dataset(f"q_{q_idx}", data=enc)
                step_f2.create_dataset(f"q_{q_idx}", data=pure_step_encodings[q_idx])


# ######### Stage 2: Check the decomposed steps
# presented = 0
# for category, answers in category_answers.items():
#     problem_steps = decomposed_answers[category]
#     for idx, answer in enumerate(answers):
#         steps = problem_steps[idx]
#         print(answer)
#         print("-" * 40)
#         print(steps)
#         print("*" * 40)
#         presented += 1
#         if presented > 5:
#             break
#     break


######### Stage 3: Plot the encodings
### Plot the encodings in the 2D spaces
plot_flag = False
step_idx = 1
while plot_flag:
    # Obtain the encodings of the current step
    cur_step_encodings, cur_step_categories, _ = result_tools.extract_category_step(
        category_encodings,
        step_idx=step_idx,
    )
    cur_step_pure_encodings, cur_step_categories, _ = (
        result_tools.extract_category_step(
            category_pure_encodings,
            step_idx=step_idx,
        )
    )

    if len(cur_step_encodings) == 0 or len(cur_step_encodings) <= 2:
        plot_flag = False
        continue

    # Plot them out
    plot_embed.plot_2d(
        encodings=cur_step_encodings,
        encoding_labels=cur_step_categories,
        label_legend="Category",
        save_path=saved_path,
        filename=f"category_step_{step_idx}_encodings_2d.png",
    )

    plot_embed.plot_2d(
        encodings=cur_step_pure_encodings,
        encoding_labels=cur_step_categories,
        label_legend="Category",
        save_path=saved_path,
        filename=f"category_step_{step_idx}_pure_encodings_2d.png",
    )
    logging.info(
        "Plotted #%d encodings in 2D space for the step %d",
        len(cur_step_encodings),
        step_idx,
    )
    step_idx += 1


######### Stage 3: Plot the relation between the question and steps

for category in list(category_pure_encodings.keys()):
    # A list of 2D arrays
    # Get the step distance distribution
    # A Dict whose key is the step index while the value is dict
    # containing, distribution and questions
    step_distance_distribution = compute_encoding.get_step_distributions(
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
    step_question_distribution = dict()
    for step_idx, value in step_distance_distribution.items():
        # Get the indexes of question has the current step
        _, _, cur_step_questions = result_tools.extract_category_step(
            category_data=category_encodings, step_idx=step_idx, category_name=category
        )
        target_q_encodings = q_encodings[cur_step_questions]
        step_question_distribution[step_idx] = compute_encoding.compute_similarity(
            target_q_encodings
        )
    plot_embed.plot_question_step_relation(
        step_distance_distribution,
        step_question_distribution,
        save_path=saved_path,
        filename=f"{category}_steps_question_relation.png",
    )


######### Stage 4: Plot the relation comparison between the question and steps
#
### Plot the encodings in the 2D spaces
plot_flag = True
step_idx = 1
while plot_flag:
    # Obtain the encodings of the current step
    cur_step_pure_encodings, cur_step_categories, cur_step_questions = (
        result_tools.extract_category_step(
            category_pure_encodings,
            step_idx=step_idx,
        )
    )

    if len(cur_step_pure_encodings) == 0:
        plot_flag = False
        continue
    distances = compute_encoding.compute_pairwise_similarity(
        encodings=cur_step_pure_encodings
    )
    plot_embed.plot_distance_heatmap(
        distances=distances,
        labels=[abbre_math_categories[cate] for cate in cur_step_categories],
        save_path=saved_path,
        filename=f"Step_{step_idx}_pure_encodings_relation",
        heatmap_config={
            "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
            "vmin": 0.4,
            "vmax": 0.8,
        },
    )

    # Get the question encodings of this step
    # cur_step_categories, cur_step_questions
    u_categories = list(dict.fromkeys(cur_step_categories))
    n_samples = len(cur_step_categories)
    step_question_encodings = []
    for category in u_categories:
        q_encodings = latent_extract.get_encodings(
            sentences=None,
            encode_method=None,
            save_path=f"{project_path}/q_sim",
            encoding_name=f"{category}_question_pure_encoding",
        )
        first_pos = cur_step_categories.index(category)
        last_pos = n_samples - cur_step_categories[::-1].index(category)
        target_positions = cur_step_questions[first_pos:last_pos]

        step_question_encodings.append(q_encodings[target_positions])

    step_question_encodings = np.vstack(step_question_encodings)
    step_q_distances = compute_encoding.compute_pairwise_similarity(
        encodings=step_question_encodings
    )

    plot_embed.plot_distance_heatmap(
        distances=step_q_distances,
        labels=[abbre_math_categories[cate] for cate in cur_step_categories],
        save_path=saved_path,
        filename=f"questions_of_step_{step_idx}_pure_encodings_relation",
        heatmap_config={
            "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
            "vmin": 0.4,
            "vmax": 0.8,
        },
    )

    step_idx += 1


# ######### Stage 4: Compute the relation between the question and steps
# metric_results = compute_encoding.computer_clustering_metrics(
#     category_step_data=category_pure_encodings
# )

# print(metric_results)
