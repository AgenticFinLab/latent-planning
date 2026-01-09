"""
The encoder and decoder of the latent plan learner.
"""

from typing import Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig

# from transformers import PreTrainedModel, PretrainedConfig
# from unsloth.chat_templates import get_chat_template

from modules.pGen_tokens import PLAN_PLACEHOLDER, RECONSTRUCT_INDICATOR

from trlm.util import template_tools


class pGenEncoder(nn.Module):
    """Encoder of the pGen module of our LpD."""

    def __init__(self, encoder_config: dict):
        """
        A encoder built with LLM is to obtain the encodings of the original textual input.
        """
        super(pGenEncoder, self).__init__()

        # Obtain the pre-trained models from the huggingface
        # as the encoders
        model_name = encoder_config["model_name"]
        self.ckpt_path = getattr(encoder_config, "ckpt_path", None)
        if self.ckpt_path is not None:
            model_name = encoder_config

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hf_config = AutoConfig.from_pretrained(model_name)

        # Get the config of the loaded encoder
        self.encoding_dim = self.hf_config.hidden_size
        self.max_seq_length = self.tokenizer.model_max_length

    def forward(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward the encoder to get token-level encodings including special tokens and general tokens.
        """
        # Tokenize sentences
        tokenized_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        # Ensure tensors are on the same device as the model
        tokenized_inputs = tokenized_inputs.to(self.model.device)
        # 1 for masked ones while 0 for the general tokens
        # Note that this is different from the attention mask,
        # tokenized_inputs["attention_mask"]), which gives
        # 1 to the general tokens, start token and end token
        # shape, (batch_size, L)
        special_tokens_mask = tokenized_inputs.pop("special_tokens_mask")
        # Compute token encodings
        model_output = self.model(**tokenized_inputs)
        # First element of model_output contains all token embeddings
        # shape, (batch_size, L, self.encoding_dim)
        token_encodings = model_output[0]

        return token_encodings, special_tokens_mask


class pGenDecoder(nn.Module):
    """Decoder of the pGen module of our LpD."""

    def __init__(self, decoder_config: dict):
        """
        A decoder-based LLM for decoding latent plans to produce the original textual input.
        """
        super(pGenDecoder, self).__init__()

        # Decoder: Autoregressive decoder-based LLM (e.g., LLaMA, GPT)
        model_name = decoder_config["model_name"]
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_config = AutoConfig.from_pretrained(model_name)

        # Get the token embedding dimension
        embedding_layer = self.llm_model.get_input_embeddings()
        self.embedding_dim = embedding_layer.weight.shape[1]
        # 2. Record the original vocabulary size.
        # Never use the vocab_size from the model as the model generally
        # define more embeddings than the vocabulary requires.
        # self.old_vocab_size = self.llm_model.config.vocab_size
        self.old_vocab_size = len(self.tokenizer)

        # The indication tokens used to indicate the start or the prompt
        # of the plan reconstruction
        self.n_indicate_tokens = decoder_config["n_indicate_tokens"]
        # Add the <IPR-{idx}> token to the decoder's tokenizer
        self.indicate_tokens = [
            RECONSTRUCT_INDICATOR.format(idx) for idx in range(self.n_indicate_tokens)
        ]
        self.placeholder = PLAN_PLACEHOLDER
        # We must add the indicate token at the end of the tokenizer
        # to allow the 'self.indicate_embeddings' to be used correctly
        self.reconstruct_special_tokens = [self.placeholder] + self.indicate_tokens

        # The newly added embeddings—including those for special tokens—are created with requires_grad=True by default.
        # This means they are trainable unless ont explicitly freezes them.
        n_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.reconstruct_special_tokens}
        )
        assert n_added == len(self.reconstruct_special_tokens)

        self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # Create a separate embedding layer for the special tokens.
        #   This embedding layer is independent and has its own name
        # Note that we must create a separate embedding layer for our
        # newly added tokens. This is because when we use the lora to the
        # finetune, we can set this separate embedding layer to be trainable to make only our newly added tokens's embeddings to be trainable.
        self.indicate_embeddings = nn.Embedding(
            len(self.indicate_tokens), self.embedding_dim
        )

        # This must be chosen carefully, so it yields 1 token each time
        self.placeholder_id = self.tokenizer.convert_tokens_to_ids(self.placeholder)
        self.indicate_ids = self.tokenizer.convert_tokens_to_ids(self.indicate_tokens)
        # Add the template to the tokenizer
        # self.tokenizer = get_chat_template(
        #     self.tokenizer,
        #     chat_template=decoder_config["chat_template"],
        #     map_eos_token=True,
        #     system_message=decoder_config["system_message"],
        # )
        # Get all special tokens (standard + additional)
        # Includes all registered special tokens
        # and Corresponding IDs
        # self.all_special_tokens = self.tokenizer.all_special_tokens
        # self.all_special_token_ids = self.tokenizer.all_special_ids
        # Get the desired special tokens of the decoder
        # For instance,
        # GPT-2 usually use <|endoftext|> the special token with ID 50256
        # These two are the desired start and end tokens indicating the text
        # input
        # self.start_token_id = self.tokenizer.bos_token_id
        self.end_token_id = self.tokenizer.eos_token_id

        self.instruction_flag, self.response_flag = template_tools.get_template_parts(
            model_name=model_name
        )
        flag_tokens = self.tokenizer.tokenize(self.response_flag)
        self.response_flag_ids = self.tokenizer.convert_tokens_to_ids(flag_tokens)

    def forward(
        self,
        post_concept_embeds: torch.Tensor,
        post_concept_attn_mask: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        num_items_in_batch: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder.

        :param input_ids: Token ids of the input string to be reconstructed
         shape, (batch_size, max_length)
        :param attention_mask: Mask of the input_ids.

        :params post_concept_embeds: Concept embeddings of shape
         (batch_size, sequence_length, concept_dim).
         Note that the name is 'post_concept_embeds' because a 'post_quant_linear' to process the actual concept embedding
         before making it as the input for the decoder.
        :params concept_attention_mask: The mask of the concept embeddings used
         to mask out the invalid or special tokens afterwards
         (batch_size, sequence_length)
        """
        d = post_concept_embeds.shape[-1]
        # We need to ensure that the input concept embedding has the same
        # dimension as the decoder's embedding
        assert d == self.embedding_dim

        ## A. Replace the embeddings of the placeholder tokens with the
        # input post_concept_embeds
        # Shape: (batch_size, max_length, self.embedding_dim)
        input_embeddings = self.llm_model.get_input_embeddings()(input_ids)

        post_concept_embeds = post_concept_embeds.to(input_embeddings.dtype)
        # Have to replace the placeholder's embeddings with the concept
        # embeddings
        B, M = input_ids.shape

        # Flatten input for vectorized operations
        # (B*M,)
        input_ids = input_ids.view(-1)
        # (B*M, d)
        input_embeddings = input_embeddings.view(-1, d)
        # (B*L,)
        post_concept_attn_mask = post_concept_attn_mask.view(-1)
        # (B*L, d)
        post_concept_embeds = post_concept_embeds.view(-1, d)

        # Identify placeholder positions in flattened input
        # shape: (#_of_placeholders,)
        pl_pos = (input_ids == self.placeholder_id).nonzero(as_tuple=True)[0]

        # Identify *active* concept positions (mask == 0)
        # shape: (#_of_active_concepts,)
        concept_pos = (post_concept_attn_mask == 0).nonzero(as_tuple=True)[0]

        # Check counts match
        if pl_pos.size(0) != concept_pos.size(0):
            raise ValueError(
                f"Number of placeholders ({pl_pos.size(0)}) "
                f"!= number of active concepts ({concept_pos.size(0)})."
            )
        # Replace placeholder embeddings with concept embeddings in one vectorized operation
        input_embeddings[pl_pos] = post_concept_embeds[concept_pos]

        ## B. Replace the special indicate tokens with the embedding layers.
        # Get the position of the indicate tokens in the input ids
        # Broadcasting: compare each element of input_ids with
        # every element in self.indicate_ids
        # Get the mask with shape (batch_size * M, )
        # True: indicate tokens, False: not indicate tokens
        # Convert self.indicate_ids to a tensor, making sure to match the dtype and device of input_ids if needed.
        indicate_ids = torch.tensor(
            self.indicate_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        is_indicate = torch.isin(input_ids, indicate_ids)

        # Get the index of the indicate ids in self.indicate_embeddings
        # We can direct minus without additionally adding -1 as the
        # input_ids start from 0
        # Shape (batch*M,)
        indicate_idxes = input_ids - indicate_ids[0]

        # Assign the indicate embeddings to the input embeddings
        # Based on the 'is_indicate'
        input_embeddings[is_indicate] = self.indicate_embeddings(
            indicate_idxes[is_indicate]
        )

        # Reshape back to (B, M, d)
        input_embeddings = input_embeddings.view(B, M, d)

        # Pass the concept embeddings through the decoder
        # input_embeddings: (batch_size, sequence_length, concept_dim)
        # Forward the decoder-only model to get the defined outputs which
        # should be dict-style format pre-defined by the decoder
        # The output will be: CausalLMOutputWithPast
        decoder_outputs = self.llm_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )
        return decoder_outputs
