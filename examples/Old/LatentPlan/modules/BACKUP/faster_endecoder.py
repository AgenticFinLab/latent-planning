"""
The encoder and decoder implemented with the unsloth of the latent plan learner.
"""

from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModel, AutoConfig

from iclp.old.util import template_tools


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
        # Supports RoPE Scaling internally, so choose any max_seq_length!
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
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
        # 1 for masked ones while o for the general tokens
        # Note that this is different from the attention mask,
        # tokenized_inputs["attention_mask"]), which gives
        # 1 to the general tokens, start token and end token
        special_tokens_mask = tokenized_inputs.pop("special_tokens_mask")
        # Compute token encodings
        model_output = self.encoder(**tokenized_inputs)
        # First element of model_output contains all token embeddings
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

        self.decoder, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name, **decoder_config["flm_config"]
        )
        self.hf_config = self.decoder.model.config

        self.decoder = FastLanguageModel.get_peft_model(
            self.decoder, **decoder_config["lora_config"]
        )

        # Get the token embedding size
        self.embedding_dim = self.decoder.get_input_embeddings().embedding_dim
        # The indication tokens used to indicate the start or the prompt
        # of the plan reconstruction
        self.n_indicate_tokens = decoder_config["n_indicate_tokens"]
        # Add the <IPR-{idx}> token to the decoder's tokenizer
        self.indicate_tokens = [f"<IPR{idx}>" for idx in range(self.n_indicate_tokens)]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.indicate_tokens}
        )
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        # This must be chosen carefully, so it yields 1 token each time
        self.placeholder = "PLAN"

        # Get the desired special tokens of the decoder
        # For instance,
        # GPT-2 usually use <|endoftext|> the special token with ID 50256
        # These two are the desired start and end tokens indicating the text
        # input
        self.start_token_id = self.tokenizer.bos_token_id
        self.end_token_id = self.tokenizer.eos_token_id

    def create_input_text(self, shape: List[int], input_strs: List[str]):
        """
        Create the input text for the decoder.
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
        formatted_text = []
        placeholder_strs = []
        input_strs = [""] * len(shape) if input_strs is None else input_strs
        for embed_length, target_str in zip(shape, input_strs):
            place_holders = " ".join([self.placeholder] * embed_length)
            reconstruct_tokens = "".join(self.indicate_tokens)
            user_content = f"{place_holders} {reconstruct_tokens}"
            if len(target_str) != 0:  # For training/finetuning
                formatted_text.append(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": target_str},
                    ]
                )
            else:  # For inference
                formatted_text.append([{"role": "user", "content": user_content}])

            placeholder_strs.append(place_holders)

        return formatted_text, placeholder_strs

    def organize_input(
        self,
        encoder_tokens_mask: torch.Tensor,
        input_strs: List[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Organize the input embeddings required by the decoder.

        :param z_q: Quantized embeddings with shape
         (batch_size, L, embedding_dim)
        :param encoder_tokens_mask: Mask matrix from the encoder side
         (batch_size, L)
        :param input_strs: The input strings.

        :return The tokenized input texts that match the target template
         of the decoder's base decoder-only model. In addition, the placeholder
         involved in the tokenized will be replaced by the concept embeddings.
        :return The start and end indexes of the placeholders. With this term, it will be easy for use to replace the embeddings of these placeholders with the concept embeddings.

        For the special decoder-only transformer, the format of the input should be in standard:

        0           1..P                P+1..P+n           P+n+1
        +---------+---------------+--------------------+-----------+
        | <start> | quantized...  | reconstruction...  |   <end>   |
        +---------+---------------+--------------------+-----------+

        A more specific example from Llama-3.2 is:
         - instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
         - response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"

         Thus, the input content (user part) should be:
            <|start_header_id|>user<|end_header_id|>\n\n ......... <|eot_id|

        For Qwen, this should be
         - instruction_part = "<|im_start|>user\n"
         - response_part = "<|im_start|>assistant\n"

         Thus, the input content (user part)  should be:
            <|im_start|>user\n ...... <|im_end|>
        """
        # Get the number of valid embeddings in the z_q
        # output a 1D tensor, shape (batch_size,)
        shape = (encoder_tokens_mask == 0).sum(dim=1)

        # Construct the input string to be the formatted text
        formatted_text, placeholders = self.create_input_text(shape, input_strs)

        # Convert the formatted text to be the input format required by
        # the large models.
        # By default, we set the add_generation_prompt to be false as in the
        # training/finetuine, we have prepare the target text which includes
        # the 'assistant' as the output
        add_generation_prompt = False
        if input_strs is None:
            # When the inference, no input_strs is set thus we have to
            # set the add_generation_prompt to be True to allow the tokenizer
            # help to add the 'assistant' to prompt the generation.
            add_generation_prompt = True  # Must add for generation
        templated_text = self.tokenizer.apply_chat_template(
            formatted_text, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        print(templated_text)

        # Perform the tokenizer,
        tokenized_text = self.tokenizer(
            templated_text,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokenized_ids = tokenized_text["input_ids"]

        # Get the position of the placeholder tokens
        placeholder_ids = self.tokenizer(placeholders, truncation=False, padding=False)[
            "input_ids"
        ]

        ## For the correctness test
        # test_ids = tokenized_ids[2]
        # for idx in range(test_ids.shape[0]):
        #     word = self.tokenizer.decode(test_ids[idx])
        #     print(idx, word)

        indices = template_tools.find_tensor(tokenized_ids, placeholder_ids)

        return tokenized_text, indices

    def forward(
        self,
        concept_embeddings: torch.Tensor,
        concept_attention_mask: torch.Tensor = None,
        input_strs: Tuple[List[str], None] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder.


        :params concept_embeddings: Concept embeddings of shape
         (batch_size, sequence_length, concept_dim).
        :params concept_attention_mask: The mask of the concept embeddings used
         to mask out the invalid or special tokens afterwards
         (batch_size, sequence_length)
        :param input_strs: The input strings that are same as the encoder part.
         When training/finetune, this is same as the input of encoder part to
         support the reconstruction.
         When inference, this is None
        """
        # Create the decoder's text input, in which we set the placeholders which are to be replaced by the concept embeddings afterwards.
        # tokenized_text:
        tokenized_text, indices = self.organize_input(
            encoder_tokens_mask=concept_attention_mask, input_strs=input_strs
        )
        # Shape: (batch_size, sequence_length, hidden_size)
        input_embeddings = self.decoder.get_input_embeddings()(
            tokenized_text["input_ids"]
        )

        # Have to replace the placeholder's embeddings with the concept
        # embeddings
        # First need to remove the special tokens from the concept embeddings
        # Original concept embedding (batch_size, L, concept_embedding_dim)
        # Given concept_attention_mask, (batch_size, L)
        batch_size = concept_embeddings.shape[0]
        for batch_i in range(batch_size):
            # Get the unmaskded B
            mask = concept_attention_mask[batch_i]
            positions = indices[batch_i]

            input_embeddings[batch_i, positions[0] : positions[1] + 1] = (
                concept_embeddings[batch_i, mask == 0]
            )

        # Pass the concept embeddings through the decoder
        # input_embeddings: (batch_size, sequence_length, concept_dim)
        decoder_outputs = self.decoder(
            inputs_embeds=input_embeddings,
            attention_mask=tokenized_text["attention_mask"],
        )
        # Get the logits directly from the decoder's output
        # Shape: (batch_size, sequence_length, vocab_size)
        logits = decoder_outputs.logits

        return logits


# Example usage
if __name__ == "__main__":

    import modules.quantizer as quantizer

    inputs = ["Hello!", "How are you?", "I am a good"]
    # Define model configuration as a dictionary
    loaded_config = {
        "encoder": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v1",
            "flm_config": {"max_seq_length": 2048, "load_in_4bit": False},
        },
        "quantizer": {
            "concept_size": 2048,
            "embedding_dim": 384,
            "beta": 0.25,
            "legacy": False,
        },
        "decoder": {
            "model_name": "Qwen/Qwen2-0.5B",
            "n_indicate_tokens": 3,
            "flm_config": {"max_seq_length": 2048, "load_in_4bit": False},
        },
    }
    define_encoder = pGenEncoder(loaded_config["encoder"])
    quantizer = quantizer.VectorQuantizer2(config=loaded_config["quantizer"])
    decoder = pGenDecoder(decoder_config=loaded_config["decoder"])

    token_encodings, special_tokens_mask = define_encoder(inputs)

    quantized_z, loss, quantized_info = quantizer(encodings=token_encodings)

    # This is for the test only
    # The input will be the concept embedding while output will have the
    # dimension same as the decoder's embedding
    linear_layer = nn.Linear(384, 896)
    quantized_z = linear_layer(quantized_z)

    output = decoder.forward(
        input_strs=inputs,
        concept_embeddings=quantized_z,
        concept_attention_mask=special_tokens_mask,
    )

    print(output)
    print(output.shape)
