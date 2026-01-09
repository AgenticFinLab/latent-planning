"""
To test the effectiveness of each part of the model.
"""

import torch
import torch.nn as nn


from modules import quantizer, endecoder

inputs = ["Hello!", "How are you?", "I am a good"]
# Define model configuration as a dictionary
loaded_config = {
    "encoder": {"model_name": "sentence-transformers/all-MiniLM-L6-v1"},
    "quantizer": {
        "concept_size": 2048,
        "embedding_dim": 384,
        "beta": 0.25,
        "legacy": False,
    },
    "decoder": {"model_name": "Qwen/Qwen2-0.5B", "n_indicate_tokens": 3},
}
# r=8,  # Rank of the low-rank matrices
# lora_alpha=16,  # Scaling factor
# target_modules=["query", "value"],  # Apply LoRA to attention layers
# lora_dropout=0.1,  # Dropout for LoRA layers
# bias="none",  # Whether to add bias
# task_type="FEATURE_EXTRACTION",  # Task type
train_config = {
    "encoder": {
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["query", "value"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "FEATURE_EXTRACTION",
        }
    },
    "decoder": {
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "FEATURE_EXTRACTION",
        }
    },
}

define_encoder = endecoder.pGenEncoder(
    loaded_config["encoder"], train_config["encoder"]["lora"]
)
defined_quantizer = quantizer.VectorQuantizer2(config=loaded_config["quantizer"])
decoder = endecoder.pGenDecoder(
    decoder_config=loaded_config["decoder"],
    lora_config=train_config["decoder"]["lora"],
)

token_encodings, special_tokens_mask = define_encoder(inputs)

quantized_z, loss, quantized_info = defined_quantizer(encodings=token_encodings)

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
