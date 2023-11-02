from typing import List, Optional, Tuple

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import torch
from torch import nn
import torch.nn.functional as F
import math
import transformers
from transformers.models.llama.modeling_llama import \
    (Union, BaseModelOutputWithPast, logger,
     LlamaConfig, LlamaDecoderLayer, LlamaRMSNorm,
     LlamaModel)


def __init__(self, config: LlamaConfig):
    super(LlamaModel, self).__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()


def move_to(x, device):
    if x is None:
        return x
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [move_to(t, device) for t in x]
    if isinstance(x, tuple):
        return tuple(move_to(t, device) for t in x)

    logger.warning_once(f'{x}')
    return x


def forward2(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    if not hasattr(self, 'model_parallel'):
        device1 = self.embed_tokens.weight.device
        if device1 == torch.device('cuda:0'):
            device2 = torch.device('cuda:1')
        else:
            device2 = torch.device('cuda:0')
        self.device2 = device2
        logger.warning_once(f'device1: {device1}')
        logger.warning_once(f'device2: {device2}')

        n_layers = len(self.layers)
        self.n_split = n_layers // 2

        for i in range(self.n_split, n_layers):
            self.layers[i] = self.layers[i].to(device2)
        self.model_parallel =True

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            device1 = input_ids.device
            device2 = self.device2

            if idx >= self.n_split:
                # logger.warning_once(f'{type(hidden_states)}')
                # logger.warning_once(f'{type(attention_mask)}')
                # logger.warning_once(f'{type(position_ids)}')
                # logger.warning_once(f'{type(past_key_values)}')


                layer_outputs = decoder_layer(
                    move_to(hidden_states, device2),
                    attention_mask=move_to(attention_mask, device2),
                    position_ids=move_to(position_ids, device2),
                    past_key_value=move_to(past_key_value, device2),
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = hidden_states.to(input_ids.device)
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def replace_llama_with_model_parallel():
    # transformers.models.llama.modeling_llama.LlamaModel.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaModel.forward = forward2
