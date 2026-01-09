"""
Methods used to define the LLM models.

For the combination of the vllm and the ray to support the distributed inference, please access the demo https://docs.vllm.ai/en/latest/getting_started/examples/distributed.html.

For the pipeline, see the https://huggingface.co/docs/transformers/zh/main_classes/pipelines.
"""

import logging
from typing import List, Any

import ray
from dotenv import load_dotenv
from litellm import batch_completion
from vllm import LLM, SamplingParams
from transformers import pipeline
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
# from projinit.config import Config

# Set LiteLLM logging level
logging.getLogger("LiteLLM").setLevel(
    logging.WARNING
    # getattr(
    #     logging,
    #     getattr(Config().logging, "LiteLLM_log_level", "WARNING").upper(),
    #     logging.WARNING,
    # )
)


class BaseLLMInference:
    """
    The basic class for the large language model based on the model directly or the vllm.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_type: str,
        generation_config: dict = None,
        vllm_config: dict = None,
    ):
        # Local or remote model name,
        # e.g., "Qwen2.5-7B" or "meta-llama/Llama-3.2-70B",
        self.model_name = model_name_or_path
        self.model_type = model_type
        self.generation_config = generation_config
        self.vllm_config = vllm_config

        self.model = None
        self.tokenizer = None

    def define_model(self):
        """Define the model and the tokenizer."""
        if self.vllm_config is not None:

            self.model = LLM(
                model=self.model_name,
                **self.vllm_config,
            )
            self.tokenizer = self.model.get_tokenizer()
        else:
            if "api" in self.model_type.lower():
                # if hasattr(Config().env, "dotenv_path"):
                #     dotenv_path = Config().env.dotenv_path
                # else:
                dotenv_path = ".env"
                load_dotenv(dotenv_path=dotenv_path)
            else:
                # Load the model and tokenizer
                self.model = pipeline(
                    task="text-generation",
                    model=self.model_name,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )

    def set_generation_config(self, generation_config: dict):
        """Set the generation configuration."""
        self.generation_config = generation_config

    def __call__(
        self,
        input_messages: List[List[dict]],
    ):
        """Inference one batch of the input prompts."""
        input_prompts: list[str] = []
        responses: list[str] = []
        # Usage of the inference, each item is a dict containing
        # "prompt_tokens", "completion_tokens", and "total_tokens".
        usages: list[dict] = []
        if self.vllm_config is not None:
            # vllm requires the input prompt to be the desired format
            # which is a message containing the system, user, and/or assistant.
            input_prompts = [
                self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in input_messages
            ]
            # Use the VLLM model for inference
            sampling_params = SamplingParams(**self.generation_config)
            outputs = self.model.generate(input_prompts, sampling_params)
            for output in outputs:
                responses.append(" ".join([o.text for o in output.outputs]))
        else:
            # Use the model with APIs for inference
            if "api" in self.model_type.lower():
                outputs = batch_completion(
                    self.model_name, messages=input_messages, **self.generation_config
                )

                for output in outputs:
                    responses.append(output["choices"][0]["message"]["content"])
                    usages.append(output["usage"])
            else:
                outputs = self.model(text=input_messages, **self.generation_config)
                for output in outputs:
                    input_prompts.append(output[0]["generated_text"])
                    responses.append(output[0]["generated_text"])

        return {
            "prompts": input_prompts,
            "responses": responses,
            "usages": usages,
        }


class RayDistributedInference:
    """
    Enable the inference performed in the distributed way.
    The inference is performed in the ray cluster.
    """

    def __init__(
        self,
        inference_operator,
        distributed_config: dict = None,
    ):
        self.inference_operator = inference_operator
        self.distributed_config = distributed_config

    def scheduling_strategy_fn(
        self,
        tensor_parallel_size: int,
        n_gpus: int = 1,
        n_cpus: int = 1,
        strategy: str = "STRICT_PACK",
    ):
        """
        Create a scheduling strategy for the given tensor parallel size.

        For tensor_parallel_size > 1, we need to create placement groups for vLLM
        to use. Every actor has to have its own placement group.
        """
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"GPU": n_gpus, "CPU": n_cpus}] * tensor_parallel_size,
            strategy=strategy,
        )
        return dict(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_capture_child_tasks=True
            )
        )

    def create_resources_kwarg(self):
        """Create the resources for the inference."""
        tensor_parallel_size = self.distributed_config.get("tensor_parallel_size", 1)
        resources_kwarg: dict[str, Any] = {}
        if tensor_parallel_size == 1:
            # For tensor_parallel_size == 1, we simply set num_gpus=1.
            resources_kwarg["num_gpus"] = 1
        else:
            # Otherwise, we have to set num_gpus=0 and provide
            # a function that will create a placement group for
            # each instance.
            resources_kwarg["num_gpus"] = 0
            resources_kwarg["ray_remote_args_fn"] = self.scheduling_strategy_fn

        return resources_kwarg

    def inference(self, dataset):
        """Inference the whole dataset."""
        # Once the distributed inference is used, we use the ray and vllm
        # by default.

        dataset = ray.data.from_huggingface(dataset)
        resources_kwarg = self.create_resources_kwarg()
        self.distributed_config.update(resources_kwarg)
        # Once the vllm is used, make the dataset to be distributed.
        # Apply batch inference for all input data.
        return dataset.map_batches(
            self.inference_operator,
            **self.distributed_config,
        )
