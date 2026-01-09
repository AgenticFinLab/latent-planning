"""
Tools used to operate the results.
"""

from typing import Dict, List, Tuple

import numpy as np


def flat_category_content(
    category_data: Dict[str, List[np.ndarray]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Flat content of categories as a ndarray.
    Here each np.ndarray in the list is a 1d vector.
    Thus, this can be used to flatten the
    1. category: questions and 2. category: answers
    """
    all_data = []
    all_labels = []
    for category, data in category_data.items():
        all_data.extend(data)
        all_labels.extend([category] * len(data))

    return np.array(all_data), all_labels


def extract_category_step(
    category_data: Dict[str, List[np.ndarray]], step_idx: int, category_name: str = None
) -> Tuple[np.ndarray, List[str]]:
    """Extract the step data of the category for the given step."""
    # Obtain the all data of the current step along all categories or the
    # required category
    cur_step_data = []
    cur_step_categories = []
    # Which questions have the desired step
    cur_step_questions = []
    category_data = (
        category_data
        if category_name is None
        else {category_name: category_data[category_name]}
    )
    for category, all_step_data in category_data.items():
        # Extract step encodings of the current step from each question
        for q_idx, step_data in enumerate(all_step_data):
            if step_idx <= len(step_data):
                cur_step_data.append(step_data[step_idx - 1])
                cur_step_categories.append(category)
                cur_step_questions.append(q_idx)

    cur_step_data = np.array(cur_step_data)
    return cur_step_data, cur_step_categories, cur_step_questions
