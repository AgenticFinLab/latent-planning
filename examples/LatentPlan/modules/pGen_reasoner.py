"""
The structure of the generator of the pGen proposed in the paper.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# from unsloth.chat_templates import get_chat_template

from modules import pGen_learner
from modules import pGen_message
from modules.pGen_tokens import (
    LATENT_PLAN_IDX,
    PLAN_START,
    PLAN_END,
    STEP_END,
    STEP_START,
    BLOCK_PLAN_START,
    BLOCK_PLAN_END,
    BLOCK_STEP_START,
    BLOCK_STEP_END,
)

import wandb


# Define a custom configuration to hold both models' configs
class ConceptGeneratorConfig(PretrainedConfig):
    def __init__(
        self,
        plan_status: str,
        generator_config: dict = None,
        learner_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Determine the status of the plan during the reasoning
        # It should be language or latent
        self.plan_status = plan_status
        self.generator_config = generator_config
        self.learner_config = learner_config

        # This is to allow the prediction part of the model
        # work well.
        # See the callback code for details.
        # Also see:
        #   https://github.com/huggingface/accelerate/issues/3277
        self.keys_to_ignore_at_inference = ["past_key_values"]


class pGenPlanReasonGenerator(PreTrainedModel):
    """
    The latent plan concept generator, which receives the question and reasoning steps already taken to output the latent plan --- the hidden
    concepts of plans.

    To build a trainable model, we are to make the codebook of the quantizer to
    be the additional tokens in the tokenizer.
    """

    config_class = ConceptGeneratorConfig

    def __init__(self, config: ConceptGeneratorConfig):
        super(pGenPlanReasonGenerator, self).__init__(config)
        self.plan_status = config.plan_status
        self.generator_config = config.generator_config
        self.learner_config = config.learner_config

        # Define the components of the generator
        model_name = self.generator_config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(model_name)
        self.hf_config = AutoConfig.from_pretrained(model_name)

        # Get the token embedding dimension
        embedding_layer = self.generator.get_input_embeddings()
        self.embedding_dim = embedding_layer.weight.shape[1]

        assert self.plan_status in ["language", "latent"]

        # The number of the latent plans in total
        # --> determined by the concept space of the learner
        self.n_latent_plans = 0
        # The strings of the latent plans
        self.latent_plan_tokens = []

        if self.plan_status == "latent":
            learner_model_config = self.learner_config["model"]
            self.n_latent_plans = learner_model_config["quantizer"]["concept_size"]
            # Extend the token
            self.latent_plan_tokens = [
                LATENT_PLAN_IDX.format(i + 1) for i in range(self.n_latent_plans)
            ]
            # Add tokens to the tokenizer
            n_added = self.tokenizer.add_tokens(self.latent_plan_tokens)
            assert n_added == self.n_latent_plans

        # Add the plan generation indicator
        self.plan_start = PLAN_START
        self.plan_end = PLAN_END
        self.step_start = STEP_START
        self.step_end = STEP_END
        self.special_tokens = [
            self.plan_start,
            self.plan_end,
            self.step_start,
            self.step_end,
        ]

        # Add the indication tokens to the tokenizer
        n_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.special_tokens}
        )
        assert n_added == 4

        # Resize embeddings (updates vocab_size automatically)
        self.generator.resize_token_embeddings(len(self.tokenizer))

        # Get the ids of our newly added tokens
        self.plan_reason_tokens = self.latent_plan_tokens + self.special_tokens
        self.plan_reason_ids = self.tokenizer.convert_tokens_to_ids(
            self.plan_reason_tokens
        )

        # Create a separate embedding layer for the newly added tokens.
        #  This embedding layer is independent and has its own name
        # Note that we must create a separate embedding layer for our
        # newly added tokens. This is because when we use the lora to the
        # finetune, we can set this separate embedding layer to be trainable to make only our newly added tokens's embeddings to be trainable.
        self.plan_embeddings = nn.Embedding(
            len(self.plan_reason_tokens), self.embedding_dim
        )
        # Get the lplan ids
        self.plan_start_id = self.tokenizer.convert_tokens_to_ids(self.plan_start)
        self.plan_end_id = self.tokenizer.convert_tokens_to_ids(self.plan_end)

        # Add the template to the tokenizer
        # self.tokenizer = get_chat_template(
        #     self.tokenizer,
        #     chat_template=self.generator_config["chat_template"],
        #     map_eos_token=True,
        #     system_message=self.generator_config["system_message"],
        # )

    def get_trainable_params_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_info = [
            (name, p.numel()) for name, p in self.named_parameters() if p.requires_grad
        ]
        total_trainable = sum(num for _, num in trainable_info)
        trainable_names = [name for name, _ in trainable_info]
        trainable_names = "; ".join(trainable_names)
        return total_params, total_trainable, trainable_names

    def get_input_embeddings(self):
        """Need to wrap for the huggingface model."""
        return self.generator.get_input_embeddings()

    def get_output_embeddings(self):
        """Need to wrap for the huggingface model."""
        return self.generator.get_output_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        num_items_in_batch: int = None,
        **kwargs,
    ):
        """Forward to generate the plan-based reasoning."""
        # Forward the token ids to get the embeddings
        # Shape: (batch_size, max_length, self.embedding_dim)
        input_embeddings = self.generator.get_input_embeddings()(input_ids)

        # Replace the embeddings for the tokens that exist in the plan_embeddings
        B, M, d = input_embeddings.shape
        # Flatten input for vectorized operations
        # (B*M,)
        input_ids = input_ids.view(-1)
        # (B*M, d)
        input_embeddings = input_embeddings.view(-1, d)
        # Get whether the token exists in the self.plan_reason_ids
        plan_reason_ids = torch.tensor(
            self.plan_reason_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        # (B*M,)
        is_new_id = torch.isin(input_ids, plan_reason_ids)

        # Get the index of the indicate ids in self.indicate_embeddings
        # We can direct minus without additionally adding -1 as the
        # input_ids start from 0
        # Shape (B*M,)
        new_indexes = input_ids - self.plan_reason_ids[0]

        # Assign the indicate embeddings to the input embeddings
        # Based on the 'is_indicate'
        input_embeddings[is_new_id] = self.plan_embeddings(new_indexes[is_new_id])

        # Reshape back to (B, M, d)
        input_embeddings = input_embeddings.view(B, M, d)

        # As the learner has been used in the 'create_plan_reason_func' to
        # process the input message by replacing the plans within the
        # self.plan_start and self.plan_start with the corresponding latent
        # plans, we can forward to get the output directly.
        return self.generator(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )

    def create_plan_reason_func(self, examples, concept_learner, is_generation=False):
        """
        Create the prompt in which the user contains the question and output is the plan-based reasoning process.

        Note that here we should replace the plans in the message's assistant to be the latent plan in the concept space learned by the learner.

        Note that each message is created by the 'organize_plan_reason_samples'
        in the `pGen_data.py`.
        """
        # A batch of messages while each message contains
        # the whole conversation
        messages = examples["message"]
        # Texts of the created input
        texts = []
        plan_reason_messages = []
        for msg in messages:
            # Add the system prompt to the msg
            if "system" not in [
                item["content"]
                for item in msg
                if isinstance(item, dict) and "role" in item
            ]:
                system_prompt = self.generator_config["system_prompt"].format(
                    self.plan_start,
                    self.plan_end,
                    self.step_start,
                    self.step_end,
                    self.plan_start,
                    self.plan_end,
                    self.step_start,
                    self.step_end,
                )
                msg.insert(0, {"role": "system", "content": system_prompt})
            # Based on the data structure, each msg contains the question and the whole plan-based reasoning process:
            # <Plan> ..... </Plan>
            # <Step> ..... </Step>

            # <Plan> ..... </Plan>
            # <Step> ..... </Step>
            # ....
            if self.plan_status == "latent":
                msg = pGen_message.create_plan_reason_message(
                    msg,
                    learner=concept_learner,
                    latent_plan_tokens=self.latent_plan_tokens,
                    start_flag=self.plan_start,
                    end_flag=self.plan_end,
                )

            template_text = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=is_generation,
            )

            texts.append(template_text)
            plan_reason_messages.append(msg)

        return {"text": texts, "plan_reason_message": plan_reason_messages}


def define_model(
    model_config,
    train_config: None,
    wandb_run: None,
    plan_status: str = "latent",
    checkpoint_path: str = None,
):
    """Define the pGen model."""

    learner_config = None
    concept_learner = None
    if plan_status == "latent":
        # Define the learner from the ckpt
        # This config will contain all config parameters of the one used to train the learner
        learner_config = model_config["learner"]
        concept_learner = pGen_learner.define_model(
            model_config=learner_config["model"],
            train_config=None,
            wandb_run=None,
            checkpoint_path=model_config["reasoner"]["learner_ckpt"],
        )
        concept_learner.eval()

        # This loop ensures that every parameter in self.learner is un-trainable. In other words, no gradients will be computed for any parameters of the learner (including those inserted by LoRA). This is necessary because—even if you set is_trainable=False—the LoRA parameters might still be marked as trainable by default.
        # Freeze all parameters in the learner module, including LoRA adapters
        for param in concept_learner.parameters():
            param.requires_grad = False

    generator_config = ConceptGeneratorConfig(
        plan_status=plan_status,
        learner_config=learner_config,
        generator_config=model_config["reasoner"],
    )

    plan_reason_generator = pGenPlanReasonGenerator(config=generator_config)
    # Log the table so far.
    init_info = plan_reason_generator.get_trainable_params_info()

    if checkpoint_path is None:
        lora_config = None if "lora" not in train_config else train_config["lora"]
        plan_reason_generator = get_peft_model(
            plan_reason_generator, peft_config=LoraConfig(**lora_config)
        )
    else:
        plan_reason_generator = PeftModel.from_pretrained(
            plan_reason_generator, checkpoint_path
        )

    lora_info = plan_reason_generator.get_trainable_params_info()

    if wandb_run is not None:
        wandb_table = wandb.Table(
            columns=[
                "Stage",
                "Total Parameters",
                "Trainable Parameters",
                "Trainable Parameter Names",
            ]
        )
        wandb_table.add_data(*["init", *init_info])
        wandb_table.add_data(*["lora", *lora_info])
        wandb_run.log({"Trainable Params Table": wandb_table})

    return plan_reason_generator, concept_learner
