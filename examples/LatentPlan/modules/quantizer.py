"""
Implementation of the quantizer of the pGen.

The core design borrows from: 
1. Taming Transformers for High-Resolution Image Synthesis.
2. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange


from transformers.utils import ModelOutput


@dataclass
class QuantizerOutput(ModelOutput):
    """
    Custom model output extending the standard CausalLMOutputWithPast
    with extra fields for your custom usage.
    """

    # Obtained quantizes with shape (batch_size, L, embedding_dim)
    z_quantizes: Optional[torch.FloatTensor] = None
    # Mask for the quantizes with shape (batch_size, L)
    # 1 for invalid ones while 0 for valid ones
    quantize_mask: Optional[torch.FloatTensor] = None
    # Loss of the quantize
    quantize_loss: Optional[torch.FloatTensor] = None
    # The corresponding indices of z_quantizes in the codebook
    # Shape (batch_size, L)
    quantize_indices: Optional[torch.FloatTensor] = None


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, config: dict):
        super(VectorQuantizer2, self).__init__()
        self.concept_size = config["concept_size"]
        self.embedding_dim = config["embedding_dim"]
        self.beta = config["beta"]

        # Set to be False by default
        self.legacy = config["legacy"]

        self.codebook = nn.Embedding(self.concept_size, self.embedding_dim)
        self.codebook.weight.data.uniform_(
            -1.0 / self.concept_size, 1.0 / self.concept_size
        )

    def forward(self, encodings: torch.Tensor, encoding_mask: torch.Tensor):
        """
        Forward to get the quantized embeddings and indices.

        :param encodings: The encodings with tensor in shape
         (batch_size, L, encoding_dim)
        :param encoding_mask: 1 for masked ones while 0 for valid ones
         ((batch_size, L)
        """
        B, L, D = encodings.shape

        # flatten -> (batch_size * L, encoding_dim)
        # To perform this operation, we should have
        # encoding_dim == self.embedding_dim
        encodings = encodings.view(-1, self.embedding_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = (
            torch.sum(encodings**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", encodings, rearrange(self.codebook.weight, "n d -> d n")
            )
        )
        # Get the embedding indices nearest to the encodings
        # Shape (batch_size * L)
        min_encoding_indices = torch.argmin(distances, dim=1)
        # Extract the corresponding latent embeddings to be the shape
        # Shape (batch_size * L, self.embedding_dim)
        z_q = self.codebook(min_encoding_indices)  # .view(encodings.shape)

        ##  Compute loss for embedding
        # Need to consider the encoding_mask
        # Shape (batch_size * L)
        encoding_mask = encoding_mask.view(-1)
        # encoding_mask == 1 => invalid, so valid positions are where mask == 0
        valid_indices = (encoding_mask == 0).nonzero(as_tuple=True)[0]

        # Filter only valid rows
        valid_z_q = z_q[valid_indices]
        valid_enc = encodings[valid_indices]

        # e.g., beta * mean(...) + mean(...)
        loss = self.beta * torch.mean(
            (valid_z_q.detach() - valid_enc) ** 2
        ) + torch.mean((valid_z_q - valid_enc.detach()) ** 2)

        # preserve gradients
        z_q = encodings + (z_q - encodings).detach()

        # Convert to the desired shape
        z_q = z_q.view(B, L, D)
        encoding_mask = encoding_mask.view(B, L)
        min_encoding_indices = min_encoding_indices.view(B, L)

        return QuantizerOutput(
            z_quantizes=z_q,
            quantize_mask=encoding_mask,
            quantize_loss=loss,
            quantize_indices=min_encoding_indices,
        )

    def get_codebook_entry(self, indices: torch.Tensor):
        """
        Get the entry from the codebook based on the indices.

        :param indices: Contain the indexes with shape
         (batch_size, L).
        """
        batch_size, L = indices.shape
        # Flatten the indices to 1d vector
        indices = indices.reshape(-1)
        # Get quantized latent embeddings
        # (batch_size * L, dim)
        z_q = self.codebook(indices)
        z_q = z_q.view(batch_size, L, -1)

        return z_q
