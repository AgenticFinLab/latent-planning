"""
This is to compute the embeddings/encodings of questions from MATH data and thus to
1. present whether they have similarity in the latent space.
"""

import os
import json
import random
from collections import defaultdict

import latent_extract
import plot_embed
import result_tools
import compute_encoding
from commons import target_math_categories, abbre_math_categories

from iclp.old.dataset import registry as dataset_registry


project_path = "examples/LatentPlan/preExperiments"

# Get all data from the MATH dataset -- training set.
trainset = dataset_registry.get("MATH", split="train")

# Randomly select 200 from each category.
category_samples = defaultdict(list)
n_selected = 200
saved_path = f"{project_path}/q_sim"
selected_filename = "selected_samples.json"
selected_samples = {}

os.makedirs(saved_path, exist_ok=True)
file_path = f"{project_path}/{selected_filename}"


if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        selected_samples = json.load(json_file)
else:
    for category, samples in category_samples.items():
        if len(samples) < n_selected:
            print(
                f"Warning: Category '{category}' has only {len(samples)} samples. Selecting all available samples."
            )
            sampled = samples.copy()
        else:
            sampled = random.sample(samples, n_selected)
        selected_samples[category] = sampled

    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(selected_samples, json_file)


selected_samples = {
    category: selected_samples[category] for category in target_math_categories
}

# Extract the features of them.
## 1. using the sentence_transformers
category_questions = {}
for category_name, sample_indexes in selected_samples.items():
    questions = [trainset[idx]["question"] for idx in sample_indexes]
    category_questions[category_name] = questions


##################################################################
######### Part 1: Get encodings with sentence_encoding ###########
##################################################################

# The original questions and the questions that are filtered with the
# stopwords
question_encodings = {}
question_pure_encodings = {}
for category_name, questions in category_questions.items():

    features = latent_extract.get_encodings(
        questions,
        encode_method=latent_extract.sentence_encoding,
        save_path=saved_path,
        encoding_name=f"{category_name}_question_encoding",
    )
    pure_features = latent_extract.get_encodings(
        latent_extract.filter_stopwords(questions),
        encode_method=latent_extract.sentence_encoding,
        save_path=saved_path,
        encoding_name=f"{category_name}_question_pure_encoding",
    )
    question_encodings[category_name] = features
    question_pure_encodings[category_name] = pure_features

## Fit and transform the embeddings
all_encodings, _ = result_tools.flat_category_content(category_data=question_encodings)
all_pure_encodings, all_categories = result_tools.flat_category_content(
    category_data=question_pure_encodings
)

## Step 2: Apply t-SNE for dimensionality reduction
plot_embed.plot_2d(
    encodings=all_encodings,
    encoding_labels=all_categories,
    label_legend="Category",
    save_path=saved_path,
    filename="encodings_2d.png",
)

plot_embed.plot_2d(
    encodings=all_pure_encodings,
    encoding_labels=all_categories,
    label_legend="Category",
    save_path=saved_path,
    filename="pure_encodings_2d.png",
)

distances, all_labels = compute_encoding.get_data_similarity(
    category_data=question_encodings
)
plot_embed.plot_distance_heatmap(
    distances=distances,
    labels=[abbre_math_categories[cate] for cate in all_labels],
    save_path=saved_path,
    filename="encodings_relation",
    heatmap_config={
        # "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
        "cbar_kws": {"ticks": [0.4, 0.8]},
        "vmin": 0.4,
        "vmax": 0.8,
    },
)

distances, all_labels = compute_encoding.get_data_similarity(
    category_data=question_pure_encodings
)
plot_embed.plot_distance_heatmap(
    distances=distances,
    labels=[abbre_math_categories[cate] for cate in all_labels],
    save_path=saved_path,
    filename="pure_encodings_relation",
    heatmap_config={
        # "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
        "cbar_kws": {"ticks": [0.4, 0.8]},
        "vmin": 0.4,
        "vmax": 0.8,
    },
)


######### Stage 4: Compute the relation between the question and steps

metric_results = compute_encoding.computer_clustering_metrics(
    category_step_data=question_pure_encodings
)

print(metric_results)
