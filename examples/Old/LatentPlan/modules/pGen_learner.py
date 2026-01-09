"""
The structure of the LpD proposed in the paper.

The design principle and core insights into the model structure borrows
from two papers:
1. Taming Transformers for High-Resolution Image Synthesis.
2. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.
"""

import re
from typing import List, Optional, Dict
from dataclasses import dataclass

import torch
from trl import SFTTrainer

# from transformers import TrainerCallback
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import quantizer
from modules import pGen_message
from modules import pGen_data
from modules.endecoder import pGenEncoder, pGenDecoder

from iclp.old.util import template_tools


@dataclass
class ConceptLearnerOutput(CausalLMOutputWithPast):
    """
    Custom model output extending the standard CausalLMOutputWithPast
    with extra fields for your custom usage.
    """

    quantize_loss: Optional[torch.FloatTensor] = None
    concept_embeddings: Optional[torch.FloatTensor] = None
    concept_embedding_mask: Optional[torch.FloatTensor] = None


# Define a custom configuration to hold both models' configs
class ConceptLearnerConfig(PretrainedConfig):
    def __init__(
        self,
        encoder_config: dict = None,
        quantizer_config: dict = None,
        decoder_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
        self.quantizer_config = quantizer_config
        self.decoder_config = decoder_config


class ConceptLearnerTrainer(SFTTrainer):
    def __init__(self, *args, concept_weight=1.0, **kwargs):
        """
        Add a constructor argument for your custom weight.
        'concept_weight' defaults to 1.0 if not provided.
        """
        super().__init__(*args, **kwargs)
        self.concept_weight = concept_weight
        self.train_loss_trace = {
            "quantize_loss": 0.0,
            "w_quantize_loss": 0.0,
            "reconstruct_loss": 0.0,
        }
        self.n_train_losses = 0

        # Trackers for evaluation phase
        self.eval_loss_trace = {
            "quantize_loss": 0.0,
            "w_quantize_loss": 0.0,
            "reconstruct_loss": 0.0,
        }
        self.n_eval_losses = 0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # We want to compute the normal next-token generation loss.
        # loss: a float value
        # outputs: ConceptLearnerOutput
        recon_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        # Compute the compound loss
        quantize_loss = outputs.quantize_loss
        w_q_loss = self.concept_weight * quantize_loss
        # Compute the weighted sum loss
        total_loss = recon_loss + w_q_loss

        # Track losses based on training/eval mode
        if model.training:
            # Training phase
            # .item() Equivalent to `detach().cpu().item()` for tensors
            self.train_loss_trace["quantize_loss"] += quantize_loss.item()
            self.train_loss_trace["w_quantize_loss"] += w_q_loss.item()
            self.train_loss_trace["reconstruct_loss"] += recon_loss.item()
            self.n_train_losses += 1
        else:
            # Evaluation phase
            self.eval_loss_trace["quantize_loss"] += quantize_loss.item()
            self.eval_loss_trace["w_quantize_loss"] += w_q_loss.item()
            self.eval_loss_trace["reconstruct_loss"] += recon_loss.item()
            self.n_eval_losses += 1

        # Update the total loss to the model loss
        outputs["loss"] = total_loss
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        """Made the log to save additional losses for different phases."""

        # Get the main key which should be
        # "loss" or "{prefix}_loss"
        pattern = re.compile(r"^(?:.*_)?loss$")
        matching_keys = [key for key in logs if pattern.match(key)]

        if len(matching_keys) != 0:
            # Get the prefix
            loss_key = matching_keys[0]
            prefix = loss_key.removesuffix("loss") if loss_key.endswith("_loss") else ""

            if self.n_train_losses > 0:
                # Calculate average training losses
                avg_train_losses = {
                    f"{prefix}{k}": v  # / self.n_train_losses
                    for k, v in self.train_loss_trace.items()
                }
                logs.update(avg_train_losses)

                # Reset training trackers
                self.train_loss_trace = {k: 0.0 for k in self.train_loss_trace}
                self.n_train_losses = 0

            if self.n_eval_losses > 0:
                # Calculate average training losses
                avg_eval_losses = {
                    f"{prefix}{k}": v / self.n_eval_losses
                    for k, v in self.eval_loss_trace.items()
                }
                logs.update(avg_eval_losses)

                # Reset training trackers
                self.eval_loss_trace = {k: 0.0 for k in self.eval_loss_trace}
                self.n_eval_losses = 0

        # Call the original logger (sends to WandB)
        super().log(logs, start_time)


# We cannot access the trainer within the callback
# class GlobalStepLossLogger(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         trainer = kwargs.get("trainer", None)
#         print("trainer: ", trainer)
#         # Check that our accumulator exists and that we've accumulated at least one loss.
#         if trainer is not None and trainer.n_losses > 0:
#             n_losses = trainer.n_losses
#             # Compute averages for each loss term.
#             avg_losses = {
#                 key: val / n_losses for key, val in trainer.loss_trace.items()
#             }
#             trainer.log(avg_losses)
#             # Reset the accumulators for the next global step.
#             trainer.loss_trace = {
#                 "quantize_loss": 0.0,
#                 "weighted_q_loss": 0.0,
#                 "reconstruct_loss": 0.0,
#             }
#             trainer.n_losses = 0

#         return control


class pGenConceptLearner(PreTrainedModel):
    """
    The latent plan concept learner, which is potentially a quantized autoencoder.
    """

    config_class = ConceptLearnerConfig

    def __init__(self, config: ConceptLearnerConfig):
        super(pGenConceptLearner, self).__init__(config)

        # Define the components of the learner
        # Encoder
        self.encoder = pGenEncoder(encoder_config=config.encoder_config)

        # Quantizer
        self.quantizer = quantizer.VectorQuantizer2(config=config.quantizer_config)

        # Decoder
        self.decoder = pGenDecoder(decoder_config=config.decoder_config)

        # The additional linear add between
        # the encoder --> linear --> quantizer
        self.prev_quant_linear = torch.nn.Linear(
            self.encoder.encoding_dim, self.quantizer.embedding_dim
        )

        # quantizer --> linear --> decoder
        self.post_quant_linear = torch.nn.Linear(
            self.quantizer.embedding_dim, self.decoder.embedding_dim
        )

        # # Set the config the our learner as the decoder's config
        # self.config = self.decoder.decoder.config

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
        return self.encoder.model.get_input_embeddings()

    def get_output_embeddings(self):
        """Need to wrap for the huggingface model."""
        return self.decoder.llm_model.get_output_embeddings()

    def encode(self, input_strs: List[str]):
        """Encode the input string to be the latent concepts."""
        encodings, encoding_mask = self.encoder(input_strs)
        encodings = self.prev_quant_linear(encodings)
        quantize_outputs = self.quantizer(encodings, encoding_mask)
        return quantize_outputs, encoding_mask

    def decode(
        self,
        quantized_z: torch.Tensor,
        quantize_mask: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ):
        """Decode latent concepts string to the text tokens."""
        quantized_z = self.post_quant_linear(quantized_z)
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            post_concept_embeds=quantized_z,
            post_concept_attn_mask=quantize_mask,
            **kwargs,
        )
        return outputs

    def decode_code(self, code_b: torch.Tensor, code_mask: torch.Tensor):
        """Decode the given codebook with the code mask."""
        quant_b = self.quantizer.get_codebook_entry(code_b)
        dec = self.decode(quantized_z=quant_b, quantize_mask=code_mask)
        return dec

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        num_items_in_batch: int = None,
        **kwargs,
    ):
        """
        Forward the whole model.

        Note that, to enable the labels are passed normally.
        We must explicitly set the labels as the argument of the function.
        """

        # Extract the ground truth plan from the response part
        # of the input ids
        # We assume that each sample only contains one
        _, contents = template_tools.get_target_indices(
            input_ids=input_ids,
            start_flag_ids=self.decoder.response_flag_ids,
            end_flag_id=self.decoder.end_token_id,
            is_return_content=True,
        )
        input_plans = [
            self.decoder.tokenizer.decode(content[0]) for content in contents
        ]

        # Encoder is to encode the pure plan strings while adding its own
        # special tokens whose positions will be identified by mask
        quant_outputs, mask = self.encode(input_strs=input_plans)
        # Post-process the quantized concept embeddings and decoder is to
        # reconstruct the original plans
        outputs = self.decode(
            quant_outputs.z_quantizes,
            quantize_mask=mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )

        return ConceptLearnerOutput(
            quantize_loss=quant_outputs.quantize_loss,
            concept_embeddings=quant_outputs.z_quantizes,
            concept_embedding_mask=mask,
            **outputs,
        )

    def create_concept_func(self, examples):
        """
        Create the concept prompt in which the user contains the placeholders.
        1. The number of the placeholders is same as the tokens that the encoder outputs from the plan.
        2. The placeholders make no sense as their embeddings will be replaced by the z_quantizes within the decoder part.
        """
        # A batch of messages while each message contains
        # the whole conversation
        messages = examples["message"]
        # Texts of the created input
        texts = []
        text_placeholders = []
        concept_messages = []
        for msg in messages:
            # Based on the data structure, each msg contains the question, previous steps and the plan
            # We need to extract the plan and create the plan message to support the reconstruct and the corresponding template text
            plan_str = pGen_data.remove_prefix(msg[-1]["content"])
            concept_message, place_holders = pGen_message.create_concept_message(
                plan_str,
                encoder_tokenizer=self.encoder.tokenizer,
                concept_placeholder=self.decoder.placeholder,
                indicate_tokens=self.decoder.indicate_tokens,
            )
            template_text = self.decoder.tokenizer.apply_chat_template(
                concept_message,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(template_text)
            text_placeholders.append(place_holders)
            concept_messages.append(concept_message)

        return {
            "text": texts,
            "placeholder": text_placeholders,
            "concept_message": concept_messages,
        }


def define_model(
    model_config, train_config: None, wandb_run: None, checkpoint_path=None
):
    """Define the pGen model."""
    learner_config = ConceptLearnerConfig(
        encoder_config=model_config["encoder"],
        quantizer_config=model_config["quantizer"],
        decoder_config=model_config["decoder"],
    )

    concept_learner = pGenConceptLearner(config=learner_config)
    # Log the table so far.
    init_info = concept_learner.get_trainable_params_info()

    if checkpoint_path is None:
        lora_config = None if "lora" not in train_config else train_config["lora"]
        concept_learner = get_peft_model(
            concept_learner, peft_config=LoraConfig(**lora_config)
        )
    else:
        concept_learner = PeftModel.from_pretrained(concept_learner, checkpoint_path)

    lora_info = concept_learner.get_trainable_params_info()

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

    return concept_learner
