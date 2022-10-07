# SPDX-License-Identifier: Apache-2.0

# TODO: now that there are two modifications to attention, i imagine a general modular attention
# should be factored out, and model-specific classes mutated to call the general one.

import math
from types import MethodType
from typing import Optional, Tuple

import torch
from torch import fft, linalg, nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaSelfAttention

from composer.algorithms.hrralibi.attention_surgery_functions.utils import (policy_registry, register_alibi,
                                                                         zero_and_freeze_expand_position_embeddings)


def hrr_binding(x, y, dim=-1):
    return fft.ifft(torch.mul(fft.fft(x, dim=dim), fft.fft(y, dim=dim)), dim=dim).real

def hrr_approx_inverse(x, dim=-1):
    return torch.roll(torch.flip(x, dims=(dim,)), 1, dims=dim)

def hrr_approx_unbinding(b, y, dim=-1):
    return hrr_binding(b, hrr_approx_inverse(y, dim=dim), dim=dim)

def hrr_exact_inverse(x, dim=-1):
    return torch.nan_to_num(fft.ifft(1. / fft.fft(x, dim=dim), dim=dim).real)

def hrr_exact_unbinding(b, y, dim=-1):
    return hrr_binding(b, hrr_exact_inverse(y, dim=dim), dim=dim)


@policy_registry.register(BertEmbeddings, RobertaEmbeddings)
def bert_embedding_converter(module: torch.nn.Module, module_index: int, max_sequence_length: int) -> torch.nn.Module:
    """Removes positional embeddings and expands `position_ids` buffer to support `max_sequence_length` tokens.
    """
    assert isinstance(module, (BertEmbeddings, RobertaEmbeddings))
    del module_index  # unused
    zero_and_freeze_expand_position_embeddings(module,
                                               max_sequence_length,
                                               position_embedding_attribute='position_embeddings')

    module_device = next(module.parameters()).device
    module.register_buffer('position_ids', torch.arange(max_sequence_length).expand((1, -1)).to(module_device))
    return module


@policy_registry.register(BertSelfAttention, RobertaSelfAttention)
def bert_attention_converter(module: torch.nn.Module, module_index: int, max_sequence_length: int) -> torch.nn.Module:
    #raise NotImplementedError("HRR for Bert ... quite yet")
    """Adds ALiBi to Bert-style SelfAttention."""
    assert isinstance(module, (BertSelfAttention, RobertaSelfAttention))
    del module_index  # unused
    module = register_alibi(module=module,
                            n_heads=int(module.num_attention_heads),
                            max_token_length=max_sequence_length,
                            causal=False)
    setattr(module, 'forward', MethodType(forward, module))

    return module


# This code is adapted from the HuggingFace Transformers library, so we ignore any type checking issues it triggers
# pyright: reportGeneralTypeIssues = none
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor]:
    """Replication of identically-named attention function function ("forward") in Composer/HuggingFace BERT model's
    BERTSelfAttention (:func:`transformers.models.bert.modeling_bert.BERTSelfAttention.forward`), but this function
    implements ALiBi and will be used to replace the default attention function."""
    mixed_query_layer = self.query(hidden_states)

    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    is_cross_attention = encoder_hidden_states is not None

    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_layer, value_layer)

    # This is the first half of the hrr modification
    #   This is the old code, frm "attention is all you need", also later
    #     # Take the dot product between "query" and "key" to get the raw attention scores.
    #     attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    #   End of old code
    # Axis names from Hrrformer paper
    T = -2
    H = -1
    # Pair keys and values using hrr binding
    attention_scores = hrr_binding(key_layer, value_layer, dim=H)
    # End first half of hrr modification

    if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
        raise NotImplementedError('ALiBi is not supported for BERT with position_embedding_type: {}'.format(
            self.position_embedding_type))
        #### REMOVES THE FOLLOWING CODE ########
        # seq_length = hidden_states.size()[1]
        # position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        # position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        # distance = position_ids_l - position_ids_r
        # positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        # positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
        #
        # if self.position_embedding_type == "relative_key":
        #     relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #     attention_scores = attention_scores + relative_position_scores
        # elif self.position_embedding_type == "relative_key_query":
        #     relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        #     relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        #     attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        ########################################

    # ALSO MAYBE REMOVED, HRR: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # This is a change to the masking arithmetic for hrr
        #   This is the old code
        #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #     attention_scores = attention_scores + attention_mask
        #   End of old code
        attention_scores = torch.where(attention_mask == 0, attention_scores, 0)
        # End of change to masking arithmetic

    ##### Modification for adding ALiBi #####
    seq_len = attention_scores.shape[-1]
    # Crop self.alibi to [1, n_heads, seq_len, seq_len]
    attention_scores = attention_scores + self.alibi[:, :, :seq_len, :seq_len]
    #########################################

    # hrr changes
    #   old code
    #     # Normalize the attention scores to probabilities.
    #     attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    #     # This is actually dropping out entire tokens to attend to, which might
    #     # seem a bit unusual, but is taken from the original Transformer paper.
    #     attention_probs = self.dropout(attention_probs)

    #     # Mask heads if we want to
    #     if head_mask is not None:
    #         attention_probs = attention_probs * head_mask
    #     
    #     context_layer = torch.matmul(attention_probs, value_layer)
    #   end of old code
    # Add the sequence elements to produce a composite representation of the terms.
    attention_scores = attention_scores.sum(dim=T)
    # Retrieve the value vectors from the query-associated keys by unbinding.
    attention_scores = hrr_approx_unbinding(attention_scores, query_layer, dim=H)
    # Calculate the final weights as the softmax of the cosine similarity.
    attention_probs = torch.nn.functional.softmax(torch.cosine_similarity(attention_scores, value_layer, dim=H))
    # Calculate the final output as the product with the original values.
    context_layer = attention_probs * value_layer
    # end of hrr changes

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs
