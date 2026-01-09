"""
This is to compute the embeddings of the answers of addressing
the questions and thus present whether there exist clusters among them.
"""

import os
import json

import latent_extract
import plot_embed
import result_tools
import compute_encoding
from commons import target_math_categories, abbre_math_categories

import numpy as np

from trlm.dataset import define_dataset


project_path = "examples/LatentPlan/preExperiments"
saved_path = f"{project_path}/answer_sim"
selected_filename = "selected_samples.json"
selected_samples = {}

os.makedirs(saved_path, exist_ok=True)
file_path = f"{project_path}/{selected_filename}"

# Get all data from the MATH dataset -- training set.
dataset = define_dataset(data_config={"data_name": "MATH"})
trainset = dataset.get_train_set()


with open(file_path, "r") as json_file:
    selected_samples = json.load(json_file)


selected_samples = {
    category: selected_samples[category] for category in target_math_categories
}


# Extract the features of them.
## 1. Get the answer of questions of different categories
category_answers = {}
for category_name, sample_indexes in selected_samples.items():
    answers = [trainset[idx]["answer"] for idx in sample_indexes]
    category_answers[category_name] = answers


answer_encodings = {}
answer_pure_encodings = {}
for category_name, answers in category_answers.items():

    features = latent_extract.get_encodings(
        answers,
        encode_method=latent_extract.sentence_encoding,
        save_path=saved_path,
        encoding_name=f"{category_name}_answer_encoding",
    )
    pure_features = latent_extract.get_encodings(
        latent_extract.filter_stopwords(answers),
        encode_method=latent_extract.sentence_encoding,
        save_path=saved_path,
        encoding_name=f"{category_name}_answer_pure_encoding",
    )
    answer_encodings[category_name] = features
    answer_pure_encodings[category_name] = pure_features


## Fit and transform the embeddings
all_encodings, _ = result_tools.flat_category_content(category_data=answer_encodings)
all_pure_encodings, all_categories = result_tools.flat_category_content(
    category_data=answer_pure_encodings
)


## Step 2: Apply t-SNE for dimensionality reduction
plot_embed.plot_2d(
    encodings=all_encodings,
    encoding_labels=all_categories,
    label_legend="Category",
    save_path=saved_path,
    filename="answer_encodings_2d.png",
)

plot_embed.plot_2d(
    encodings=all_pure_encodings,
    encoding_labels=all_categories,
    label_legend="Category",
    save_path=saved_path,
    filename="answer_pure_encodings_2d.png",
)

distances, all_labels = compute_encoding.get_data_similarity(
    category_data=answer_encodings
)
plot_embed.plot_distance_heatmap(
    distances=distances,
    labels=[abbre_math_categories[cate] for cate in all_categories],
    save_path=saved_path,
    filename="answer_encodings_relation",
    heatmap_config={
        "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
        "vmin": 0.4,
        "vmax": 0.8,
    },
)

distances, all_labels = compute_encoding.get_data_similarity(
    category_data=answer_pure_encodings
)
plot_embed.plot_distance_heatmap(
    distances=distances,
    labels=[abbre_math_categories[cate] for cate in all_categories],
    save_path=saved_path,
    filename="answer_pure_encodings_relation",
    heatmap_config={
        "cbar_kws": {"label": "Cosine Distance", "ticks": [0.4, 0.8]},
        "vmin": 0.4,
        "vmax": 0.8,
    },
)


######### Stage 4: Compute the relation between the question and steps

metric_results = compute_encoding.computer_clustering_metrics(
    category_step_data=answer_pure_encodings
)

print(metric_results)
