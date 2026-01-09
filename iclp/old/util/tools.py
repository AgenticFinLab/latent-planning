"""Tools used by the whole project."""

import re
from typing import List
from collections import Counter


def format_term(terminology: str):
    """Format the terminology to be the standard one.
    This function ensure that all the terminology are in the same format.
    """
    # basic conversion
    terminology = (
        terminology.replace("_", " ").replace("and", "&").replace("-", " ").rstrip()
    )
    if terminology.isupper():
        return terminology

    return terminology.title()


def compare_steps_answer(steps: List[str], answer: str):
    """
    Compare the differences between the splitted steps and the given answer.
    """
    # Define a regex pattern to remove the "Step X:" prefix
    pattern = r"^Step \d+: "

    # Apply the regex pattern to remove "Step X:" and combine the strings
    combined_string = " ".join([re.sub(pattern, "", s).strip() for s in steps])

    # Count the characters and words in both strings
    combined_char_count = Counter(combined_string)
    paragraph_char_count = Counter(answer)

    # Find character differences
    char_differences = (
        combined_char_count
        - paragraph_char_count
        + paragraph_char_count
        - combined_char_count
    )
    total_char_differences = sum(char_differences.values())
    combined_word_count = Counter(combined_string.split())
    paragraph_word_count = Counter(answer.split())

    # Find word differences
    word_differences = (
        combined_word_count
        - paragraph_word_count
        + paragraph_word_count
        - combined_word_count
    )
    total_word_differences = sum(word_differences.values())

    return total_char_differences, total_word_differences
