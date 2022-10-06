# SPDX-License-Identifier: Apache-2.0

# TODO: now that there are two modifications to attention, i imagine a general modular attention
# should be factored out, and model-specific classes mutated to call the general one.

from types import MethodType
from typing import Tuple

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model

from composer.algorithms.alibi.attention_surgery_functions.utils import (policy_registry, register_alibi,
                                                                         zero_and_freeze_expand_position_embeddings)

from torch.fft import fft, ifft, fft2, ifft2

def hrr_approx_inverse(x):
    return torch.roll(torch.flip(x, dims=(-1,)), 1, dims=-1)

def hrr_exact_inverse(x):
    return torch.nan_to_num(ifft(1. / fft(x)).real)

def hrr_inverse_2d(x):
    return torch.nan_to_num(ifft2(1. / fft2(x)).real)

def hrr_projection(x):
    return torch.nan_to_num(ifft(fft(x) / torch.abs(fft(x))).real)

def hrr_projection_2d(x):
    return torch.nan_to_num(ifft2(fft2(x) / torch.abs(fft2(x))).real)

def hrr_binding(x, y):
    return ifft(torch.mul(fft(x), fft(y))).real

def hrr_binding_2d(x, y):
    return ifft2(torch.mul(fft2(x), fft2(y))).real

def hrr_approx_unbinding(b, y):
    return hrr_binding(b, hrr_approx_inverse(y))

def hrr_exact_unbinding(b, y):
    return hrr_binding(b, hrr_exact_inverse(y))

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
    setattr(module, '_attn', MethodType(_attn, module))

    module = enlarge_mask(module, max_sequence_length)
    return module


def _attn(self, query, key, value, attention_mask=None, head_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:


    # This is the first half of the hrr modification
    #   This is the old code, prior to the modification
    #     attn_weights = torch.matmul(query, key.transpose(-1, -2))
    #   End of old code
    # Pair keys and values using hrr binding
    attn_weights = hrr_binding_2d(key.transpose(-1, -2), value.transpose(-1, -2)).transpose(-1, -2)
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

    assert not self.is_cross_attention
    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

    # ====> i'm here. beta is attn_weights.transpose(-1,-2).
    # i'm checking whether attention_mask uses large values, since they are summed
    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.Softmax(dim=-1)(attn_weights)
    attn_weights = self.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

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
