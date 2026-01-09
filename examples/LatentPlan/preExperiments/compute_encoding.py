"""
Compute the statistics relation of the encodings.
"""

from typing import Dict, List, Tuple, Union

import result_tools

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def compute_similarity(encodings: np.ndarray) -> np.ndarray:
    """Compute the similarity among encodings."""
    # Get the central as (1, n_dims)
    central = np.mean(encodings, axis=0).reshape(1, -1)
    # Return the array with shape (n_encodings,)
    return cdist(encodings, central, metric="cosine").flatten()


def compute_pairwise_similarity(encodings: np.ndarray) -> np.ndarray:
    """Compute the similarity among encodings."""
    return cdist(encodings, encodings, metric="cosine")


def get_category_centroids(
    category_data: Dict[str, List[np.ndarray]]
) -> Dict[str, np.ndarray]:
    """
    Get the centroids of categories between categories.
    """
    return {
        category: np.mean(np.stack(vectors), axis=0)
        for category, vectors in category_data.items()
    }


def get_category_similarity(
    category_data: Dict[str, List[np.ndarray]], metric="cosine"
):
    """
    Compute the similarity between categories.
    To measure the similarity, we need to first compute the mean of one term, such as the questions or the answers, of categories.
    the categories.
    """
    category_centroids = get_category_centroids(category_data)

    center_matrix = np.array([category_centroids[cat] for cat in category_centroids])

    # 2) Use scipy's cdist() to compute pairwise distances (Euclidean by default) als cosine
    dist_mat = cdist(center_matrix, center_matrix, metric=metric)
    return dist_mat


def get_data_similarity(
    category_data: Dict[str, List[np.ndarray]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the similarity between items of the categories.
    This function is generally used to compute the similarity between the questions or answers of categories.
    """
    all_data, all_labels = result_tools.flat_category_content(category_data)
    # 2) Use scipy's cdist() to compute pairwise distances (Euclidean by default) als cosine
    dist_mat = cdist(all_data, all_data, metric="cosine")

    return dist_mat, all_labels


def get_step_distributions(
    category_step_data: Dict[str, List[np.ndarray]],
    category_name: str,
) -> Dict[int, Dict[str, Union[np.ndarray, list]]]:
    """
    Get the distance distribution of the step-wise data of the category.

    This is generally used to present the relation:
    1. Between the questions and each reasoning step of the category.
    2. Between the questions and each reasoning plan of the category.
    """
    visit_flag = True
    step_idx = 1
    category_step_distances = {}
    while visit_flag:
        cur_step_data, _, _ = result_tools.extract_category_step(
            category_step_data, step_idx=step_idx, category_name=category_name
        )
        if len(cur_step_data) == 0:
            visit_flag = False
            continue
        # Compute the distance distribution

        central = np.mean(cur_step_data, axis=0).reshape(1, -1)
        distance_distribution = 1 - cdist(cur_step_data, central, metric="cosine")
        # Save the corresponding category with a flattened 1D array
        category_step_distances[step_idx] = distance_distribution.flatten()
        step_idx += 1

    return category_step_distances


def get_question_distributions(
    category_data: Dict[str, List[np.ndarray]], category_name: str = None
):
    """
    Get the distance distribution of the step-wise data of the category.
    """
    # A list of arrays
    q_encodings = category_data[category_name]
    central = np.mean(q_encodings, axis=0).reshape(1, -1)
    return cdist(q_encodings, central, metric="cosine")


def computer_clustering_metrics(
    category_step_data: Dict[str, List[np.ndarray]]
) -> Dict[str, float]:
    """Compute necessary metrics to present the step clustering information."""
    all_data, all_labels = result_tools.flat_category_content(
        category_data=category_step_data
    )

    # 3. Compute cluster validity metrics
    # +1: clear separation, 0: overlapping clusters (on average)
    # -1: misclassified or reversed clusters
    # A higher silhouette indicates better-defined clusters.
    # Scores range from -1 (poor clustering) to 1 (perfect clustering).
    sil_score = silhouette_score(all_data, all_labels)
    # Measures ratio of between-cluster dispersion to within-cluster dispersion
    # Higher is better
    ch_score = calinski_harabasz_score(all_data, all_labels)
    # Davies-Bouldin Score (DB)
    # Measures average “similarity” (ratio of within-cluster scatter to between-cluster separation)
    # Lower is better
    db_score = davies_bouldin_score(all_data, all_labels)

    return {
        "Silhouette_Score": sil_score,
        "Calinski-Harabasz": ch_score,
        "Davies-Bouldin": db_score,
    }
