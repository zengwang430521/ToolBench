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
    (Union, LlamaModel, LlamaForCausalLM,
     CausalLMOutputWithPast, CrossEntropyLoss)


def __init__(self, config):
    super(LlamaForCausalLM, self).__init__(config)

    # student model without thought
    self.model = LlamaModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # teacher model with thought
    self.model_teacher = LlamaModel(config)
    self.lm_head_teacher = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Initialize weights and apply final processing
    self.post_init()
    self.lambda_dist = 1.0
    self.lambda_teacher = 1.0
    self.lambda_student = 1.0


def loss_distillation_feature(x_student, x_teacher, move_idx_feature, thought_mask, labels):
    # debug
    x_gt = torch.gather(x_teacher.detach(), dim=1, index=move_idx_feature[:, :, None].expand_as(x_teacher))[:, :-1]
    x_student = x_student[:, :-1]

    # 是需要预测的部分，而且不是thought部分，才给loss
    mask = (labels > 0) * (~thought_mask)
    mask = mask[:, 1:, None]
    loss = F.l1_loss(x_student*mask, x_gt*mask, reduction='mean')
    return loss


def loss_distillation_simple(x_student, x_teacher, thought_mask, labels):
    # debug
    x_teacher = x_teacher[:, :-1].detach()
    x_student = x_student[:, :-1]

    # 下一个单词是action，才给loss才给loss
    mask = (labels > 0) * (~thought_mask)
    mask = mask[:, 1:, None]
    loss = F.l1_loss(x_student*mask, x_teacher*mask, reduction='mean')
    return loss


def forward2(
        self,
        input_ids: torch.LongTensor = None,
        thought_mask: Optional[torch.Tensor] = None,
        move_idxs_label: Optional[torch.Tensor] = None,
        move_idxs_feature: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    # Example:
    #
    # ```python
    # >>> from transformers import AutoTokenizer, LlamaForCausalLM
    #
    # >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    # >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
    #
    # >>> prompt = "Hey, are you consciours? Can you talk to me?"
    # >>> inputs = tokenizer(prompt, return_tensors="pt")
    #
    # >>> # Generate
    # >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    # >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
    # ```
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict


    '''
    student model, causal mask + mask out thought tokens
    '''
    B, L = input_ids.shape
    # attn_mask_student = torch.ones(L, L, dtype=torch.bool).tril(diagonal=0) # causal
    attn_mask_student = input_ids.new_ones(L, L, dtype=torch.bool).tril(diagonal=0) # causal
    attn_mask_student = attn_mask_student[None, ...].expand(B, L, L)
    attn_mask_student = attn_mask_student.masked_fill(thought_mask[:, None, :], False)
    # import matplotlib.pyplot as plt
    # plt.imshow(attn_mask_student[0].detach().cpu().numpy())
    # import cv2
    # cv2.imwrite('attn.png', attn_mask_student[0].detach().float().cpu().numpy()*255)

    '''因为删除了thought token， 第1个action token需要提前'''
    labels_student = labels.masked_fill(thought_mask, -100)
    labels_student = torch.gather(labels_student, dim=1, index=move_idxs_label)

    # # for debug
    # labels_student = torch.arange(100)[None, :].expand([2, 100]) / 100.0
    # move_idxs_label = torch.arange(100)[None, :].repeat([2, 1])
    # move_idxs_label[0] = 99 - move_idxs_label[0]
    # labels_student = torch.gather(labels_student, dim=1, index=move_idxs_label)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attn_mask_student,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        # output_hidden_states=output_hidden_states,
        output_hidden_states=True,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]
    hidden_features = outputs.hidden_states
    logits = self.lm_head(hidden_states)

    # print('student')

    loss = 0
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_student[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_student = loss_fct(shift_logits, shift_labels) * self.lambda_student
        loss = loss + loss_student

    '''
    teacher model: causal mask (realized in decoder layer) 
    '''
    attn_mask_teacher = input_ids.new_ones(L, L, dtype=torch.bool).tril(diagonal=0) # causal
    attn_mask_teacher = attn_mask_teacher[None, ...].expand(B, L, L)

    outputs_teacher = self.model_teacher(
        input_ids=input_ids,
        attention_mask=attn_mask_teacher,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        # output_hidden_states=output_hidden_states,
        output_hidden_states=True,
        return_dict=return_dict,
    )
    hidden_states_teacher = outputs_teacher[0]
    hidden_features_teacher = outputs_teacher.hidden_states
    logits_teacher = self.lm_head_teacher(hidden_states_teacher)

    # print('teacher')

    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits_teacher[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_teacher = loss_fct(shift_logits, shift_labels) * self.lambda_teacher
        loss = loss + loss_teacher

    '''distillation loss'''
    # print('distillation')
    loss_dist = 0
    for x_s, x_t in zip(hidden_features, hidden_features_teacher):
        # loss_dist += loss_distillation_feature(x_s, x_t, move_idxs_feature, thought_mask, labels)
        loss_dist += loss_distillation_simple(x_s, x_t, thought_mask, labels) * self.lambda_dist

    loss = loss +  loss_dist

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # print('loss', loss)
    # print('loss_student', loss_student)
    # print('loss_teacher', loss_teacher)
    # print('loss_distillation', loss_dist)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )



def replace_llama_with_thought():
    # transformers.models.llama.modeling_llama.LlamaModel.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward2
    transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ = __init__
