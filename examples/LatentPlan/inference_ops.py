"""
All function required to perform the inference.
"""

from typing import List, NamedTuple

import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer_pt_utils import EvalLoopContainer
from transformers.tokenization_utils import PreTrainedTokenizer

# Use a TextStreamer for continuous inference - so you can see the generation token by token, instead of waiting the whole time!
from transformers import TextStreamer

from trlm.dataset import define_dataset
from trlm.prompt import plan_prompt
from projinit.config import Config
from trlm.util import template_tools


class EvalLoopInputOutputWith(NamedTuple):
    input_ids: np.ndarray
    input_strs: List[str]

    ori_label_ids: np.ndarray
    label_ids: np.ndarray
    label_strs: List[str]
    logits: np.ndarray
    predict_texts: List[str]

    def to_csv_dict(self):
        """Convert the data to be format savable by the csv file."""
        assert (
            self.input_ids.shape[0] == len(self.input_strs)
            and len(self.input_strs) == self.ori_label_ids.shape[0]
            and self.ori_label_ids.shape[0] == self.label_ids.shape[0]
            and self.label_ids.shape[0] == len(self.label_strs)
            and len(self.label_strs) == self.logits.shape[0]
            and self.logits.shape[0] == len(self.predict_texts)
        )

        return {
            "input_ids": self.input_ids.tolist(),
            "input_strs": self.input_strs,
            "ori_label_ids": self.ori_label_ids.tolist(),
            "label_ids": self.label_ids.tolist(),
            "label_strs": self.label_strs,
            "logits": self.logits.tolist(),
            "predict_texts": self.predict_texts,
        }


def decode_predictions(
    tokenizer: PreTrainedTokenizer, predictions: EvalLoopOutput
) -> dict:
    """
    Decodes predictions and labels from the model output.

    Args:
        tokenizer: The tokenizer used for decoding.
        predictions: The output from trainer.predict(), which contains both
                     .predictions and .label_ids.
    Returns:
        A dictionary with decoded 'labels' and 'predictions'.
    """
    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_strs = tokenizer.batch_decode(
        predictions.label_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    # Get the index with the highest logit for each token.
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {
        "ori_label_ids": predictions.label_ids,
        "label_ids": label_ids,
        "label_strs": label_strs,
        "logits": logits,
        "predict_texts": prediction_text,
    }


def inference_with_trainer(
    trainer: Trainer,
    dataset: Dataset,
):
    """Inference the reason on the input samples of the model."""

    predictions = trainer.predict(dataset)
    all_input_ids = EvalLoopContainer(
        trainer.args.eval_do_concat_batches, padding_index=-100
    )
    input_strs = []
    dataloader = trainer.get_test_dataloader(dataset)
    for _, inputs in enumerate(dataloader):
        all_input_ids.add(inputs["input_ids"])
        strs = trainer.tokenizer.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        input_strs += strs

    all_input_ids = all_input_ids.get_arrays()
    decoded_outputs = decode_predictions(
        tokenizer=trainer.tokenizer, predictions=predictions
    )

    return EvalLoopInputOutputWith(
        input_ids=all_input_ids, input_strs=input_strs, **decoded_outputs
    )
