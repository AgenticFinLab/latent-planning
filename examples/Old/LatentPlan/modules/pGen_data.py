"""
Transform the dataset to support the model fine-tuning on the plan generation.

After downloading the data from the huggingface, we have the datasets containing samples with the format presented as `PlanReasonSample` in `synthetic/synthetic_pipeline.py`. It mainly has
    - question, steps, explicit_plans, skeleton_plans

Then, we are able to create the following types of plan-based reasoning process for the model training.

1. Single-Round Plan-based Reasoning (SR)
    For one question, all plans are given at once within the tag `<plans> ... </plans>` and thus the reasoning process can be obtained based on these plans. This process can be presented as follows:
        Question: ...
        <plans>
        ...
        </plans>
        <steps>
        ...
        </steps>

    By using different plans, we have the following two categories:

    1.1 Using explicit plans -> naming `SRExplicitPRSample`
        When the explicit plans are used, we create the sample `ExplicitPlanReasoningSample` in which the plans within the `<plans>` are specific instructions.

    1.2 Using skeleton plans -> naming `SRSkeletonPRSample`
        When the skeleton plans are used, we create the sample `SkeletonPlanReasoningSample` in which the plans within the `<plans>` are skeleton instructions.

    1.3 Using latent plans -> naming `SRLatentPRSamp
        When guiding the reasoning with latent plans, we first need to use `pGen_learner` to get the latent concept token of the `skeleton_plans`
        and thus place these tokens within the `<plans>` block as the guidance.


2. Multiple-Round Plan-based Reasoning
    For one question, the plan and reasoning step appear interactively, meaning that each plan is followed by a reasoning step. The format is:
        Question: ...
        Plan 1:  ...\n\n
        Step 1: ...\n
        Plan 2:... \n\n
        Step 2: ... \n
        ....

    Based on the types of the plan used in the above format, we also have three types just like the 1.1, 1.2, and 1.3.
"""

import re
import logging
from typing import Dict, List
from dataclasses import dataclass

from transformers.utils import ModelOutput as FieldFrozenContainer

from modules.pGen_tokens import PLAN_START, PLAN_END, STEP_START, STEP_END


def extract_message(message: List[Dict[str, str]]):
    """Get the content from a message."""
    msg_authors = []
    msg_content = []
    for item in message:
        msg_authors.append(item["role"])
        msg_content.append(item["content"])

    return msg_authors, msg_content


def create_plan_prompt(
    question: str,
    steps: List[str],
):
    """Create the prompt for the plan generation."""
    user_plan_prompt = """\n\n{}{}\n\nPlease generate the Plan {} for guiding the next reasoning step {}."""
    n_steps = len(steps)
    head = f"{question}" if n_steps == 0 else f"{question}\n\n"
    return user_plan_prompt.format(
        head,
        "\n".join(steps),
        n_steps + 1,
        n_steps + 1,
    )


def create_plan_reasoning_process(
    plans: List[str], steps: List[str], stop_plan: str
) -> str:
    """Create the reasoning process containing the plan and the step."""
    # Combine the strings from A and B.
    process_str = "\n\n".join(
        f"{PLAN_START} {p} {PLAN_END}\n{STEP_START} {s} {STEP_END}"
        for p, s in zip(plans, steps)
    )

    # Append the 'stop' string at the end, with two newlines as a separator.
    process_str += "\n\n" + stop_plan

    return process_str


def stop_plan():
    """Get the plan to stop the reasoning."""
    return "Complete reasoning with no additional steps required."


def create_reasoning_prompt(question: str, steps: List[str], plan: str):
    """Create the prompt to perform the reasoning with the guidance of plan."""
    user_plan_prompt = """\n\n{}{}\n\n{}\n\nFollowing the high-level idea in Plan {} to generate the next reasoning Step {}."""
    n_steps = len(steps)
    head = f"{question}" if n_steps == 0 else f"{question}\n\n"
    return user_plan_prompt.format(
        head,
        "\n".join(steps),
        plan,
        n_steps,
        n_steps,
        n_steps + 1,
    )


def remove_prefix(input_str: str, prefix: str = "Plan"):
    """
    Removes prefixes like:
      - 'Plan:'
      - 'Plan 5:'
      - '*Plan something*:'  (with or without spaces)
    from the beginning of the string.
    """
    assert prefix in ["Plan", "Step"]
    # Explanation of the pattern:
    # ^          - start of string
    # \*?        - match 0 or 1 '*' (e.g., '*Plan')
    # Plan       - literal 'Plan'
    # [^:]*      - zero or more characters that are not ':'
    # :          - literal colon
    # \s*        - zero or more whitespace characters
    if prefix == "Plan":
        prefix_pattern = r"\s*\*?Plan[^:]*:\s*"
    else:
        prefix_pattern = r"\s*\*?Step[^:]*:\s*"
    input_str = re.sub(prefix_pattern, "", input_str)
    # Step 2: Remove any leftover leading asterisks and whitespace
    #  ^[\*\s]+    - one or more '*' or whitespace at start
    input_str = re.sub(r"^[\*\s]+", "", input_str)

    return input_str


def organize_plan_samples(dataset, plan_name: str = "Plans"):
    """Organize samples for the plan generation."""
    n_samples = len(dataset)
    logging.info("Organizing %s samples from %d samples", plan_name, n_samples)
    plan_samples = []

    for idx in range(n_samples):
        # 'IDLabler', 'source', 'timestamp', 'discipline', 'category', 'sub-field', 'question', 'Steps', 'Plans'
        solution_sample = dataset[idx]
        source = solution_sample["source"]
        question = solution_sample["question"]
        steps = solution_sample["Steps"]
        plans = solution_sample[plan_name]
        n_steps = len(steps)

        # Add the plan samples gradually
        for idx in range(n_steps):
            plan_str = remove_prefix(plans[idx], prefix="Plan")
            plan_samples.append(
                PlanSample(
                    IDLabler=solution_sample["IDLabler"],
                    source=f"{source}-Plan{idx+1}",
                    message=[
                        {
                            "role": "user",
                            "content": create_plan_prompt(question, steps=steps[:idx]),
                        },
                        {
                            "role": "assistant",
                            "content": f"Plan {idx + 1}: {plan_str}",
                        },
                    ],
                )
            )
        # Add the final plan as the reasoning stop
        end_stop = n_steps + 1
        plan_samples.append(
            PlanSample(
                IDLabler=solution_sample["IDLabler"],
                source=f"{source}-Plan{end_stop}",
                message=[
                    {
                        "role": "user",
                        "content": create_plan_prompt(question, steps=steps),
                    },
                    {
                        "role": "assistant",
                        "content": stop_plan(),
                    },
                ],
            )
        )
    logging.info("Organized %d %s samples", len(plan_samples), plan_name)
    return plan_samples


def organize_plan_reason_samples(dataset, plan_name: str = "Plans"):
    """Organize samples for the plan-based reasoning."""
    n_samples = len(dataset)
    logging.info("Organizing %s samples from %d samples", plan_name, n_samples)

    plan_samples = []

    for idx in range(n_samples):
        # 'IDLabler', 'source', 'timestamp', 'discipline', 'category', 'sub-field', 'question', 'Steps', 'Plans'
        solution_sample = dataset[idx]
        source = solution_sample["source"]
        question = solution_sample["question"]
        steps = solution_sample["Steps"]
        plans = solution_sample[plan_name]
        pure_plans = [remove_prefix(p, prefix="Plan") for p in plans]
        pure_steps = [remove_prefix(s, prefix="Step") for s in steps]
        plan_samples.append(
            PlanSample(
                IDLabler=solution_sample["IDLabler"],
                source=f"{source}-PlanReasoning{idx+1}",
                message=[
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\n",
                    },
                    {
                        "role": "assistant",
                        "content": create_plan_reasoning_process(
                            plans=pure_plans, steps=pure_steps, stop_plan=stop_plan()
                        ),
                    },
                ],
            )
        )

    logging.info("Organized %d %s samples", len(plan_samples), plan_name)
    return plan_samples
