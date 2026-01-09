"""
Get the hidden plan from the question and the reasoning steps.
"""

import logging


import torch
from unsloth import FastLanguageModel


class LatentPlanTransformer(torch.nn.Module):
    """The encoder to extract the latent plan."""

    def __init__(self, model_config: dict, train_config: dict):
        """ """
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        self.lpe_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config["model_name"],
            max_seq_length=model_config["max_seq_length"],
            dtype=None,
            load_in_4bit=model_config["load_in_4bit"],
        )
        self.old_vocab_size = len(self.tokenizer)
        ## Tunable Part
        # The size of the triggers used to cooperate with the question and
        # steps to produce the plan
        self.trigger_size = self.model_config.plan_trigger_size
        # We set the trigger tokens' word as follows
        tmlp = "[PT{}]"
        self.PTRIGGER_TOKENS = [
            tmlp.format(idx + 1) for idx in range(self.trigger_size)
        ]
        num_added_tokens = self.tokenizer.add_tokens(self.PTRIGGER_TOKENS)
        logging.info(
            "Added #%d plan triggers, leading to #%d Added tokens.",
            self.trigger_size,
            num_added_tokens,
        )
        assert num_added_tokens == self.trigger_size
        # Increase the vocabulary size to be vocab_size + trigger_size
        #  so that the trigger tokens are in
        #  [self.vocab_size, self.vocab_size + self.mem_size)
        self.vocab_size_with_trigger = self.old_vocab_size + self.trigger_size
        # Create the embeddings of the table
        self.lpe_model.resize_token_embeddings(self.vocab_size_with_trigger)
        logging.info(
            "Increased vocabulary from %d to %d.",
            self.old_vocab_size,
            self.vocab_size_with_trigger,
        )

        ## Special Tokens
        # Added after the trigger_size to start the latent plan generation
        self.TRIGGER_END_TOKEN = "[PTE]"
        # Added after the latent plan for the reasoning start
        self.PLAN_FOR_REASONING = "[LPR]"
        num_added_tokens = self.tokenizer.add_tokens(
            [self.TRIGGER_END_TOKEN, self.PLAN_FOR_REASONING]
        )
        logging.info(
            "Added #2 special tokens, leading to #%d Added tokens.",
            num_added_tokens,
        )
        assert num_added_tokens == self.trigger_size
        self.new_vocab_size_with = self.vocab_size_with_trigger + 2
        self.lpe_model.resize_token_embeddings(self.new_vocab_size_with)
        logging.info(
            "Increased vocabulary from %d to %d.",
            self.vocab_size_with_trigger,
            self.new_vocab_size_with,
        )

        ## Next we need to operate the embedding layers
        # We follow the example in the line 340 of https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
        # Collect the new tokens
        self.new_tokens = self.PTRIGGER_TOKENS.extend(
            [self.TRIGGER_END_TOKEN, self.PLAN_FOR_REASONING]
        )

        ## Freeze all parameters except for the embeddings of the new tokens
        self.num_embeddings = len(self.lpe_model.get_input_embeddings().weight.data)

        # Get the new embedding indices
        new_token_ids = self.tokenizer.convert_tokens_to_ids(self.new_tokens)
        new_embeddings = self.lpe_model.get_input_embeddings().weight.data[
            new_token_ids
        ]

        # Unfreeze the new token embeddings
        for idx in range(new_token_ids):
            if idx in new_token_ids:
                self.lpe_model.get_input_embeddings().weight.data[idx].requires_grad = True
            else:
