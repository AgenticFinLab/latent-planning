"""
Using LLMs to synthesize data from the samples.
"""

from typing import List

from synthetic import synthetic_prompt
from synthetic import utils as synthetic_utils

from trlm.dataset import base
from trlm.model.llms import BaseLLMInference


def decompose_message_prompt(question: str, answer: str):
    """
    Create the prompt to decompose the answer into logical reasoning steps.
    """

    prompt = f"""Question: {question}\n\nAnswer: {answer}\n\n{synthetic_prompt.PlanPrompts.answer_decompose_prompt}"""

    return [
        {
            "role": "system",
            "content": synthetic_prompt.PlanSystemPrompts.answer_decompose_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def explicit_plan_message_prompt(question: str, steps: str):
    """
    Create the prompt to summarize the explicit plans from steps.
    """

    prompt = f"""Question: {question}\nSteps:{steps}\n\n{synthetic_prompt.PlanPrompts.explicit_plan_summary_prompt}"""

    return [
        {
            "role": "system",
            "content": synthetic_prompt.PlanSystemPrompts.explicit_plan_summary_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def skeleton_plan_message_prompt(question: str, steps: str):
    """
    Create the prompt to summarize the explicit plans from steps.
    """

    prompt = f"""Question: {question}\nSteps:{steps}\n\n{synthetic_prompt.PlanPrompts.skeleton_plan_summary_prompt}"""

    return [
        {
            "role": "system",
            "content": synthetic_prompt.PlanSystemPrompts.skeleton_plan_summary_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


class DirectPlanSynthesize:
    """Directly synthesize the data from the samples."""

    def __init__(
        self,
        decompose_step_config: dict,
        explicit_plan_config: dict,
        skeleton_plan_config: dict,
    ):
        # Generation config
        # for the answer decomposition
        self.decompose_step_config = decompose_step_config
        # for the explicit plan summarization
        self.explicit_plan_config = explicit_plan_config
        # for the skeleton plan summarization
        self.skeleton_plan_config = skeleton_plan_config

        # When these parts use the same llm, we can only
        # create one llm to be shared and change the generation
        # config and prompts for different tasks.
        # Create an LLM.
        # We force the same model currently.
        self.inference_model = BaseLLMInference(
            model_name_or_path=self.decompose_step_config["model_name"],
            model_type=self.decompose_step_config["model_type"],
            generation_config=self.decompose_step_config["generation_config"],
            vllm_config=self.decompose_step_config.get("vllm_config", None),
        )
        self.inference_model.define_model()

    def __call__(self, batch_samples: List[base.TextSample]):
        """Inference the sample to obtain the synthesized data."""

        message_prompts = [
            decompose_message_prompt(sample["question"], sample["cot_answer"])
            for sample in batch_samples
        ]

        outputs = self.inference_model(input_messages=message_prompts)

        is_matched = synthetic_utils.check_match(
            [sample["cot_answer"] for sample in batch_samples],
            outputs["responses"],
        )
        # Get the decomposed steps that is one string
        batch_steps = outputs["responses"]

        # Summarize explicit plans from the decomposed steps
        self.inference_model.set_generation_config(self.explicit_plan_config)
        message_prompts = [
            explicit_plan_message_prompt(sample["question"], steps)
            for sample, steps in zip(batch_samples, batch_steps)
        ]
        outputs = self.inference_model(input_messages=message_prompts)
        # Get the explicit plans that is one string
        explicit_plans = outputs["responses"]

        # Summarize skeleton plans from the decomposed steps
        self.inference_model.set_generation_config(self.skeleton_plan_config)
        format_prompts = [
            skeleton_plan_message_prompt(sample["question"], steps)
            for sample, steps in zip(batch_samples, batch_steps)
        ]
        outputs = self.inference_model(format_prompts)
        # Get the skeleton plans that is one string
        skeleton_plans = outputs["responses"]
        assert len(skeleton_plans) == len(batch_steps)

        return {
            "steps": batch_steps,
            "explicit_plans": explicit_plans,
            "skeleton_plans": skeleton_plans,
            "matched": is_matched,
        }
