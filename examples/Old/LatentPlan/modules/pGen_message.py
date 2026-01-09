"""
The tokenizer used by the pGen.
"""

import re
from typing import List


from transformers import PreTrainedTokenizer


def create_concept_message(
    input_str: str,
    encoder_tokenizer: PreTrainedTokenizer,
    concept_placeholder: str,
    indicate_tokens: List[str],
):
    """
    Create the input message for the decoder.
    This function aims to simplify the 'organize_input()' by creating
    #batch_size new texts while each text is:
        When training/finetune:
        [
            {"role": "user",
            "content": "<embedding_length> #<special_tokens>"},
            {"role": "assistant",
            "content": input_ids}}
        ]
        When inference:
        [
            {"role": "user",
            "content": "<embedding_length> #<special_tokens>"}
        ]

    More specific, the content of the user should be:
                1..P                        P+1..P+n
    -----------------------------+--------------------------------+
    | quantized placeholders...  | reconstruction indications...  |
    +----------------------------+--------------------------------+
    """
    formatted_message = ""
    # Get the concept length, which should be obtained from the encoder's
    # tokenizer
    # Determine how many placeholders are required.
    tokenized_ids = encoder_tokenizer(
        input_str,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_special_tokens_mask=False,
    )["input_ids"]

    concept_len = len(tokenized_ids)

    place_holders = " ".join([concept_placeholder] * concept_len)
    reconstruct_tokens = "".join(indicate_tokens)
    user_content = f"{place_holders} {reconstruct_tokens}"
    if len(input_str) != 0:  # For training/finetuning
        formatted_message = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": input_str},
        ]

    else:  # For inference
        formatted_message = [{"role": "user", "content": user_content}]

    return formatted_message, place_holders


# We need to ensure how many placeholders are required.
def create_plan_reason_message(
    message: List[dict],
    learner: PreTrainedTokenizer,
    latent_plan_tokens: List[str],
    start_flag: str,
    end_flag: str,
) -> str:
    """
    Create the message for the plan-based reason process.
    One important of this function is to replace the string plans with the
    latent plans.

    The message is:
      [
       {"role": "user",      "content": "..."},
       {"role": "assistant", "content": "..."},
       ...
      ]

    """
    user_question = [item["content"] for item in message if item["role"] == "user"][0]
    reasoning_str = [
        item["content"] for item in message if item["role"] == "assistant"
    ][0]

    created_message = []

    # Regex pattern to capture everything between start_flag and end_flag
    start_esc = re.escape(start_flag)
    end_esc = re.escape(end_flag)

    # Build the pattern dynamically
    # '(.*?)' is the non-greedy capture group for content between flags
    pattern = re.compile(rf"{start_esc}(.*?){end_esc}", re.DOTALL)

    plans = []
    plan_positions = []
    for match in pattern.finditer(reasoning_str):
        # group(1) is the content inside <Plan>...</Plan>
        content = match.group(1)

        # By default, match.start() and match.end() give the positions
        # of the entire match (<Plan> + content + </Plan>).
        # If we only want the positions of the content itself,
        # we use match.start(1) and match.end(1).
        content_start = match.start(1)
        content_end = match.end(1)

        plans.append(content)
        plan_positions.append((content_start, content_end))

    N = len(plans)

    if N != 0:  # training or finetuning
        # Assume that we have N plans, i.e., N-1 steps, required in the reasoning
        # process, where the 1 means the fixed plan representing the completion.
        # 'encoding_mask' here presents the validation of the lplans
        # Shape, (N, L)
        quantize_outputs, encoding_mask = learner.encode(input_strs=plans)
        # Shape, (N+1, L) where L is the max length of the learner's encoder
        indices = quantize_outputs.quantize_indices

        # Obtain the defined tokens of the indices obtained from the learner
        latent_indices = [indices[i][encoding_mask[i] == 0] for i in range(N)]
        # Obtain the tokens of the latent plans obtained from the learner
        latent_tokens = [
            " ".join([latent_plan_tokens[ind] for ind in latent_indices[i]])
            for i in range(N)
        ]

        # Replace the plan contents in the reasoning process with the latent plan
        # tokens
        # Process positions and replacements in reverse order
        # to avoid messing up the indices of subsequent replacements.
        for (start, end), repl in zip(
            reversed(plan_positions), reversed(latent_tokens)
        ):
            reasoning_str = reasoning_str[:start] + repl + reasoning_str[end:]

    created_message = [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": reasoning_str},
    ]

    return created_message
