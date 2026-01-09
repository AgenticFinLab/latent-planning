"""
This file is to extract the 
"""

import os
from typing import List, Callable, Any

import nltk
import torch
import numpy as np
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


def filter_stopwords(input_sentences: List[str]):
    """Filter the stop words in the input sentence."""
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    filtered_sentences = []
    for sentence in input_sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        new_sentence = " ".join(filtered_words)
        filtered_sentences.append(new_sentence)

    return filtered_sentences


def sentence_encoding(input_sentences: List[str]):
    """Get the encodings of the sentences."""
    # Initialize the model
    # all-MiniLM-L6-v2: Balances speed and accuracy; suitable for many similarity tasks. - 384 dimension
    # all-mpnet-base-v2: Higher accuracy but larger size; ideal when precision is paramount.
    # paraphrase-MiniLM-L6-v2: Specifically fine-tuned for paraphrase identification
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    # Get the tensor with shape
    # [num_sentences, dimension]
    embeddings = model.encode(input_sentences)

    return embeddings


def sentence_llm_encoding(
    input_sentences: List[str],
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    type: str = "last",
):
    """Get the encodings of the sentence by using the LLMs."""
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # # Tokenize the sentences
    # inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
    # The above operations are equal to the following one
    # Initialize the feature extraction pipeline
    feature_extractor = pipeline("feature-extraction", model=model_name)
    # Extract features without specifying max_length
    # By default, it processes each sequence individually and returns a list of feature representations:
    # features: a list where each element corresponds to a sequence from the input list
    features = feature_extractor(input_sentences)

    if type == "mean":
        # Aggregate to get sentence-level features for each sentence
        # Example: Take the mean of all token embeddings per sentence
        # A list of tensors
        encodings = [torch.mean(feature, dim=0) for feature in features]
    elif type == "last":
        # Use the feature of the last token
        # Tokenize the sequences to get the attention mask
        inputs = feature_extractor.tokenizer(
            input_sentences, return_tensors="pt", padding=True, truncation=True
        )
        attention_mask = inputs["attention_mask"]

        # Convert features to tensor
        features_tensor = [
            torch.tensor(f) for f in features
        ]  # Convert each sequence's embeddings into tensors

        # Find the last non-padding token for each sequence
        last_token_indices = (
            attention_mask.sum(dim=1) - 1
        )  # Subtract 1 for 0-based indexing

        # Extract last token encodings
        encodings = [
            features_tensor[i][last_token_indices[i]]
            for i in range(len(features_tensor))
        ]
    else:
        # [CLS]: Recommended for tasks where the [CLS] token has been pre-trained to summarize the sequence.
        # Extract the [CLS] token embedding (first token in each sequence)
        encodings = [feature[0] for feature in features]

    return encodings


def get_encodings(
    sentences: List[str],
    encode_method: Callable[..., Any],
    save_path: str,
    encoding_name: str,
):
    """
    Get the encodings of the sentences.
    :return encodings: An array containing the encodings of the input.
    """
    save_file_path = f"{save_path}/{encoding_name}.npz"

    if os.path.exists(save_file_path):
        encodings = np.load(save_file_path)["arr_0"]
    else:
        encodings = encode_method(input_sentences=sentences)
        np.savez(save_file_path, encodings)

    return encodings
