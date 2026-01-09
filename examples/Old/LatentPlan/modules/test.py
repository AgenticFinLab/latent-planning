"""
This test is to verify whether out implementations of the embeddings replacement
used in the decoder makes sense --- differentiable.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a Hugging Face decoder-only transformer (e.g., Qwen)
model_name = "Qwen/Qwen2-0.5B"  # Replace with the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
decoder = AutoModelForCausalLM.from_pretrained(model_name)


# Define the encoder (custom or pre-trained)
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        return hidden


# Example inputs
inputs = ["Hello!", "How are you?", "I am good"]

# Tokenize inputs
tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
input_ids = tokenized_inputs["input_ids"]  # (batch_size, sequence_length)
attention_mask = tokenized_inputs["attention_mask"]  # (batch_size, sequence_length)

# Concept embeddings and mask (example)
batch_size, seq_length = input_ids.shape
embedding_dim = decoder.config.hidden_size  # Use the model's hidden size
concept_embeddings = torch.randn(
    batch_size, seq_length, embedding_dim
)  # (batch_size, L, embedding_dim)
concept_attention_mask = torch.tensor(
    [[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]]
)  # (batch_size, L)
indices = torch.tensor(
    [[1, 2], [0, 1], [2, 4]]
)  # (batch_size, 2) - start and end positions


# Forward pass
def forward(input_ids, concept_embeddings, concept_attention_mask, indices):
    # Get input embeddings from the decoder
    input_embeddings = decoder.get_input_embeddings()(
        input_ids
    )  # (batch_size, seq_length, embedding_dim)

    # Replace placeholder embeddings with concept embeddings
    batch_size = concept_embeddings.shape[0]
    for batch_i in range(batch_size):
        mask = concept_attention_mask[batch_i]
        start, end = indices[batch_i]
        print(mask)
        print(concept_embeddings[batch_i].shape)
        # Get unmasked concept embeddings
        unmasked_concept_embeddings = concept_embeddings[batch_i, mask == 0]

        # Replace embeddings in input
        input_embeddings[batch_i, start : end + 1] = unmasked_concept_embeddings

    # Pass the modified embeddings through the decoder
    outputs = decoder(inputs_embeds=input_embeddings, attention_mask=attention_mask)
    logits = outputs.logits  # (batch_size, seq_length, vocab_size)
    return logits


# Compute loss and gradients
logits = forward(input_ids, concept_embeddings, concept_attention_mask, indices)

# Example targets (next token prediction)
targets = input_ids[:, 1:].contiguous()  # Shift input_ids to get targets
logits = logits[:, :-1, :].contiguous()  # Align logits with targets

# Compute loss
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
loss.backward()

# Check gradients
print("Loss:", loss.item())
print("Gradients for decoder:", decoder.get_input_embeddings().weight.grad)
print("Gradients for encoder:", encoder.get_input_embeddings().weight.grad)
