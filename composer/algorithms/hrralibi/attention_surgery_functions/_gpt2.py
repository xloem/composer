# SPDX-License-Identifier: Apache-2.0

# TODO: now that there are two modifications to attention, i imagine a general modular attention
# should be factored out, and model-specific classes mutated to call the general one.

from types import MethodType
from typing import Tuple

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model

from composer.algorithms.hrralibi.attention_surgery_functions.utils import (policy_registry, register_alibi,
                                                                         zero_and_freeze_expand_position_embeddings)

from torch.fft import fft, ifft, fft2, ifft2
from torch.linalg import vector_norm as norm

def hrr_approx_inverse(x, dim=-1):
    return torch.roll(torch.flip(x, dims=(dim,)), 1, dims=dim)

def hrr_exact_inverse(x, dim=-1):
    return torch.nan_to_num(ifft(1. / fft(x, dim=dim), dim=dim).real)

def hrr_inverse_2d(x):
    return torch.nan_to_num(ifft2(1. / fft2(x)).real)

def hrr_projection(x, dim=-1):
    return torch.nan_to_num(ifft(fft(x, dim=dim) / torch.abs(fft(x, dim=dim)), dim=dim).real)

def hrr_projection_2d(x):
    return torch.nan_to_num(ifft2(fft2(x) / torch.abs(fft2(x))).real)

def hrr_binding(x, y, dim=-1):
    return ifft(torch.mul(fft(x, dim=dim), fft(y, dim=dim)), dim=dim).real

def hrr_binding_2d(x, y):
    return ifft2(torch.mul(fft2(x), fft2(y))).real

def hrr_approx_unbinding(b, y, dim=-1):
    return hrr_binding(b, hrr_approx_inverse(y, dim=dim), dim=dim)

def hrr_exact_unbinding(b, y, dim=-1):
    return hrr_binding(b, hrr_exact_inverse(y, dim=dim), dim=dim)

def hrr_unbinding_2d(b, y):
    return hrr_binding_2d(b, hrr_inverse_2d(y))


@policy_registry.register(GPT2Model)
def gpt2_embedding_converter(module: torch.nn.Module, module_index: int, max_sequence_length: int) -> torch.nn.Module:
    """Removes positional embeddings."""
    assert isinstance(module, GPT2Model)
    del module_index  # unused

    zero_and_freeze_expand_position_embeddings(module, max_sequence_length, position_embedding_attribute='wpe')
    return module


@policy_registry.register(GPT2Attention)
def gpt2_attention_converter(module: torch.nn.Module, module_index: int, max_sequence_length: int) -> torch.nn.Module:
    """Adds ALiBi to GPT2Attention and replaces the attention mask to support `max_sequence_length` tokens."""

    assert isinstance(module, GPT2Attention)
    del module_index  # unused
    module = register_alibi(module=module,
                            n_heads=int(module.num_heads),
                            max_token_length=max_sequence_length,
                            causal=True)

    # Alibi
    setattr(module, '_attn', MethodType(_attn, module))

    module = enlarge_mask(module, max_sequence_length)

    # HRR
    module.bias = module.bias.bool() # causal mask
    del module.masked_bias # attention mask

    return module


def _attn(self, query, key, value, attention_mask=None, head_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:

    # axis labels T and H are from the paper
    T = -2 # sequence
    H = -1 # embeddings
    # -3 is heads


    # This is the first half of the hrr modification
    #   This is the old code, frm "attention is all you need"
    #     attn_weights = torch.matmul(query, key.transpose(-1, -2))
    #   End of old code
    # Pair keys and values using hrr binding
    attn_weights = hrr_binding(key, value, dim=H)
    # End first half of hrr modification

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1))**0.5)

    # This is the alibi modification
    n_tokens = attn_weights.shape[-1]
    # Truncate alibi distance weights to size of current batch
    alibi = self.alibi[:, :, 0:n_tokens]
    # alibi = self.alibi[:, :, :, 0:n_tokens].repeat(batch_size, 1, 1, 1)
    attn_weights = attn_weights + alibi
    # End alibi modification

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)

        # This is a change to the masking arithmetic for hrr
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        attn_weights = torch.where(causal_mask, attn_weights, 0)
        # End change to masking arithmetic


    if attention_mask is not None:
        # Apply the attention mask

        # This is a change to the masking arithmetic for hrr
        # The broadcast comparison could be optimized into GPT2Model.forward .
        attn_weights = torch.where(attention_mask == 0, attn_weights, 0)
        # End change to masking arithmetic

    # This old code was here but its effect is replaced by the HRR attention.
    #  attn_weights = torch.nn.Softmax(dim=-1)(attn_weights)
    # End of old code

    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # This is the second half of the hrr modification,
    #   This is the old code
    #     attn_output = torch.matmul(attn_weights, value)
    #   End of old code
    # Add the sequence elements to produce a composite representation of the terms.
    attn_weights = attn_weights.sum(dim=T)
    # Retrieve the value vectors from the query-associated keys by unbinding.
    attn_weights = hrr_approx_unbinding(attn_weights, query, dim=H)
    # Calculate the final weights as the softmax of the cosine similarity.
    attn_weights = torch.nn.functional.softmax(torch.cosine_similarity(attn_weights, value, dim=H))
    # Calculate the final output as the product with the original values.
    attn_output = attn_wieghts * value
    # End second half of hrr modification

    # ====> i've gone through this function once and attempted to convert it.
    #       i have not checked for typos or parse failures or anything or tried running it.
    #       some of the utility functions at the top could be removed.

    return attn_output, attn_weights


def enlarge_mask(module: torch.nn.Module, max_sequence_length: int) -> torch.nn.Module:
    """Increases the size of the attention mask in Composer/HuggingFace GPT2 model's GPT2Attention
    (:func:`transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn`; `GitHub link <https://\\
    github.com/huggingface/transformers/blob/2e11a043374a6229ec129a4765ee4ba7517832b9/src/transformers/\\
    models/gpt2/modeling_gpt2.py#L140>`_).

    This is necessary for evaluating on sequence lengths longer than the model was initialized to accommodate.
    """
    old_mask = module.bias
    new_mask = torch.tril(
        torch.ones(
            (max_sequence_length, max_sequence_length),  # type: ignore
            dtype=torch.uint8,
            device=old_mask.device)).view(1, 1, max_sequence_length, max_sequence_length)  # type: ignore
    setattr(module, 'bias', new_mask)
    return module
