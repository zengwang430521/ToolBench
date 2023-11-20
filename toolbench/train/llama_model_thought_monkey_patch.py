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
from transformers.modeling_utils import *
from transformers.modeling_utils import (
    _add_variant, _load_state_dict_into_model,
    _load_state_dict_into_meta_model, )
import gc
import os
import re
import shutil
import tempfile
# import pdb

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
    # import pdb
    # pdb.set_trace()
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
    # # print('distillation')
    # loss_dist = 0
    # for x_s, x_t in zip(hidden_features, hidden_features_teacher):
    #     # loss_dist += loss_distillation_feature(x_s, x_t, move_idxs_feature, thought_mask, labels)
    #     loss_dist += loss_distillation_simple(x_s, x_t, thought_mask, labels) * self.lambda_dist
    #
    # loss = loss + loss_dist

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


@classmethod
def _load_pretrained_model2(
    cls,
    model,
    state_dict,
    loaded_keys,
    resolved_archive_file,
    pretrained_model_name_or_path,
    ignore_mismatched_sizes=False,
    sharded_metadata=None,
    _fast_init=True,
    low_cpu_mem_usage=False,
    device_map=None,
    offload_folder=None,
    offload_state_dict=None,
    dtype=None,
    load_in_8bit=False,
    keep_in_fp32_modules=None,
):
    # pdb.set_trace()

    # print('state_dict')
    # print(state_dict)
    # print('pretrained_model_name_or_path')
    # print(pretrained_model_name_or_path)
    # print('loaded keys')
    # print(loaded_keys)

    '''不用调整model的key, 把loaded key, state_dict 更新就行'''
    loaded_keys += [t.replace('model.', 'model_teacher.').replace('lm_head.', 'lm_head_teacher.') for t in loaded_keys]


    is_safetensors = False
    if load_in_8bit:
        from transformers.utils.bitsandbytes import set_module_8bit_tensor_to_device

    if device_map is not None and "disk" in device_map.values():
        archive_file = (
            resolved_archive_file[0] if isinstance(resolved_archive_file, (list, tuple)) else resolved_archive_file
        )
        is_safetensors = archive_file.endswith(".safetensors")
        if offload_folder is None and not is_safetensors:
            raise ValueError(
                "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                " offers the weights in this format."
            )
        if offload_folder is not None:
            os.makedirs(offload_folder, exist_ok=True)
        if offload_state_dict is None:
            offload_state_dict = True

    is_sharded_safetensors = is_safetensors and sharded_metadata is not None
    # Retrieve missing & unexpected_keys
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    prefix = model.base_model_prefix

    def _fix_key(key):
        if "beta" in key:
            return key.replace("beta", "bias")
        if "gamma" in key:
            return key.replace("gamma", "weight")
        return key

    original_loaded_keys = loaded_keys
    loaded_keys = [_fix_key(key) for key in loaded_keys]

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    # key re-naming operations are never done on the keys
    # that are loaded, but always on the keys of the newly initialized model
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module  # False
    add_prefix_to_model = has_prefix_module and not expects_prefix_module       # False

    if remove_prefix_from_model:
        _prefix = f"{prefix}."
        expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
        expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
        # '''model_teacher'''
        # expected_keys_not_prefixed = []
        # expected_keys_new = []
        # prefix2 = 'model_teacher'
        # _prefix = f"{prefix}."
        # _prefix2 = f"{prefix2}."
        # for s in expected_keys:
        #     if s.startswith(_prefix):
        #         expected_keys_new.append(s[len(_prefix):])
        #     elif s.startswith(_prefix2):
        #         expected_keys_new.append(s[len(_prefix2):])
        #     else:
        #         expected_keys.append(s)
        #         expected_keys_not_prefixed.append(s)
        # expected_keys = expected_keys_new
    elif add_prefix_to_model:
        expected_keys = [".".join([prefix, s]) for s in expected_keys]

    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))

    # Some tensors maybe have been already filled by another key (tied weights).
    existing_ptrs = {model_state_dict[k].data_ptr() for k in loaded_keys if k in model_state_dict}
    missing_keys = [
        k for k in missing_keys if k in model_state_dict and model_state_dict[k].data_ptr() not in existing_ptrs
    ]
    # Some models may have keys that are not in the state by design, removing them before needlessly warning
    # the user.
    if cls._keys_to_ignore_on_load_missing is not None:
        for pat in cls._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if cls._keys_to_ignore_on_load_unexpected is not None:
        for pat in cls._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    # retrieve weights on meta device and put them back on CPU.
    # This is not ideal in terms of memory, but if we don't do that not, we can't initialize them in the next step
    if low_cpu_mem_usage:
        for key in missing_keys:
            if key in list(model_state_dict.keys()):
                key = key
            elif f"{prefix}.key" in list(model_state_dict.keys()):
                key = f"{prefix}.key"
            elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(model_state_dict.keys()):
                key = ".".join(key.split(".")[1:])
            param = model_state_dict[key]

            # upcast in fp32 if any
            target_dtype = dtype
            if (
                keep_in_fp32_modules is not None
                and dtype == torch.float16
                and any(module_to_keep_in_fp32 in key for module_to_keep_in_fp32 in keep_in_fp32_modules)
            ):
                target_dtype = torch.float32

            if param.device == torch.device("meta"):
                if not load_in_8bit:
                    set_module_tensor_to_device(model, key, "cpu", torch.empty(*param.size(), dtype=target_dtype))
                else:
                    set_module_8bit_tensor_to_device(
                        model, key, "cpu", torch.empty(*param.size(), dtype=target_dtype)
                    )

    # retrieve unintialized modules and initialize before maybe overriding that with the pretrained weights.
    if _fast_init:
        if remove_prefix_from_model:
            _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
        elif add_prefix_to_model:
            _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
        else:
            _loaded_keys = loaded_keys
        set_initialized_submodules(model, _loaded_keys)
        # This will only initialize submodules that are not marked as initialized by the line above.
        model.apply(model._initialize_weights)

    # Set some modules to fp32 if any
    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                param = param.to(torch.float32)

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = model
    # print(len(cls.base_model_prefix))
    # print(hasattr(model, cls.base_model_prefix))
    # print(has_prefix_module)

    if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
        start_prefix = cls.base_model_prefix + "."
    if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
        model_to_load = getattr(model, cls.base_model_prefix)
        base_model_expected_keys = list(model_to_load.state_dict().keys())
        if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
            raise ValueError(
                "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                "properly saved?"
            )
        if device_map is not None:
            device_map = {k.replace(f"{cls.base_model_prefix}.", ""): v for k, v in device_map.items()}

    def _find_mismatched_keys(
        state_dict,
        model_state_dict,
        loaded_keys,
        add_prefix_to_model,
        remove_prefix_from_model,
        ignore_mismatched_sizes,
    ):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key
                if remove_prefix_from_model:
                    # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                    model_key = f"{prefix}.{checkpoint_key}"
                elif add_prefix_to_model:
                    # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                    model_key = ".".join(checkpoint_key.split(".")[1:])

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                ):
                    mismatched_keys.append(
                        (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                    )
                    del state_dict[checkpoint_key]
        return mismatched_keys

    if resolved_archive_file is not None:
        folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
    else:
        folder = None
    if device_map is not None and is_safetensors:
        param_device_map = expand_device_map(device_map, original_loaded_keys)

        str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
        if sharded_metadata is None:
            archive_file = (
                resolved_archive_file[0]
                if isinstance(resolved_archive_file, (list, tuple))
                else resolved_archive_file
            )
            weight_map = {p: archive_file for p in original_loaded_keys}
        else:
            weight_map = {p: os.path.join(folder, f) for p, f in sharded_metadata["weight_map"].items()}
        offload_index = {
            p: {"safetensors_file": f, "weight_name": p, "dtype": str_dtype}
            for p, f in weight_map.items()
            if param_device_map[p] == "disk"
        }

    if state_dict is not None:
        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            original_loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )
        error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
        offload_index = None

        # '''model teacher'''
        # print('load teacher model')
        # error_msgs += _load_state_dict_into_model(model.model_teacher, state_dict, start_prefix)

    else:
        # Sharded checkpoint or whole but low_cpu_mem_usage==True

        # This should always be a list but, just to be sure.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]

        error_msgs = []
        mismatched_keys = []
        if not is_safetensors:
            offload_index = {} if device_map is not None and "disk" in device_map.values() else None
        if offload_state_dict:
            state_dict_folder = tempfile.mkdtemp()
            state_dict_index = {}
        else:
            state_dict_folder = None
            state_dict_index = None

        if is_sharded_safetensors:
            disk_only_shard_files = get_disk_only_shard_files(device_map, sharded_metadata=sharded_metadata)
            disk_only_shard_files = [os.path.join(folder, f) for f in disk_only_shard_files]
        else:
            disk_only_shard_files = []

        if len(resolved_archive_file) > 1:
            resolved_archive_file = logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")
        for shard_file in resolved_archive_file:
            # Skip the load for shards that only contain disk-offloaded weights when using safetensors for the offload.
            if shard_file in disk_only_shard_files:
                continue
            state_dict = load_state_dict(shard_file)

            '''在这里复制key'''
            for key in list(state_dict.keys()):
                key_new = key.replace('model.', 'model_teacher.').replace('lm_head.', 'lm_head_teacher.')
                state_dict[key_new] = state_dict[key]

            # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
            # matching the weights in the model.
            mismatched_keys += _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            if low_cpu_mem_usage:
                # '''model teacher'''
                # print('load student')
                # print('start_prefix', start_prefix)
                # print(type(model))
                # print(type(model_to_load))
                # print(list(state_dict.keys())[0])

                new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    loaded_keys,
                    start_prefix,
                    expected_keys,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    offload_index=offload_index,
                    state_dict_folder=state_dict_folder,
                    state_dict_index=state_dict_index,
                    dtype=dtype,
                    load_in_8bit=load_in_8bit,
                    is_safetensors=is_safetensors,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                )
                error_msgs += new_error_msgs

                # '''model teacher'''
                # print('load teacher')
                # print('start_prefix', start_prefix)
                # state_dict_new = {}
                # for key in state_dict:
                #     key_new = key[len('model.'):]
                #     state_dict_new[key_new] = state_dict[key]
                #
                # print(type(model))
                # print(list(state_dict_new.keys())[0])
                # print(list(model.model_teacher.state_dict().keys())[0])
                #
                # new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                #     model.model_teacher,
                #     state_dict_new,
                #     # loaded_keys,
                #     [],
                #     start_prefix,
                #     expected_keys,
                #     device_map=device_map,
                #     offload_folder=offload_folder,
                #     offload_index=offload_index,
                #     state_dict_folder=state_dict_folder,
                #     state_dict_index=state_dict_index,
                #     dtype=dtype,
                #     load_in_8bit=load_in_8bit,
                #     is_safetensors=is_safetensors,
                #     keep_in_fp32_modules=keep_in_fp32_modules,
                # )
                # error_msgs += new_error_msgs

            else:
                error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
                # '''model teacher'''
                # print('load teacher')
                # error_msgs += _load_state_dict_into_model(model.model_teacher, state_dict, start_prefix)

            # force memory release
            del state_dict
            gc.collect()

        if offload_index is not None and len(offload_index) > 0:
            if model != model_to_load:
                # We need to add the prefix of the base model
                prefix = cls.base_model_prefix
                if not is_safetensors:
                    for weight_name in offload_index:
                        shutil.move(
                            os.path.join(offload_folder, f"{weight_name}.dat"),
                            os.path.join(offload_folder, f"{prefix}.{weight_name}.dat"),
                        )
                offload_index = {f"{prefix}.{key}": value for key, value in offload_index.items()}
            if not is_safetensors:
                save_offload_index(offload_index, offload_folder)
                offload_index = None

        if offload_state_dict:
            # Load back temporarily offloaded state dict
            load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
            shutil.rmtree(state_dict_folder)

    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    if load_in_8bit:
        unexpected_keys = [elem for elem in unexpected_keys if "SCB" not in elem]
        missing_keys = [elem for elem in missing_keys if "SCB" not in elem]

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
            " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
            f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
            " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
    else:
        logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    elif len(mismatched_keys) == 0:
        logger.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
            f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
            " training."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )

    return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs


@classmethod
def from_pretrained2(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
    r"""
    Instantiate a pretrained pytorch model from a pre-trained model configuration.

    The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
    the model, you should first set it back in training mode with `model.train()`.

    The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
    pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
    task.

    The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
    weights are discarded.

    Parameters:
        pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing model weights saved using
                  [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                  this case, `from_tf` should be set to `True` and a configuration object should be provided as
                  `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                  PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                  `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                  `True`.
                - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                  arguments `config` and `state_dict`).
        model_args (sequence of positional arguments, *optional*):
            All remaining positional arguments will be passed to the underlying model's `__init__` method.
        config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
            Can be either:

                - an instance of a class derived from [`PretrainedConfig`],
                - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

            Configuration for the model to use instead of an automatically loaded configuration. Configuration can
            be automatically loaded when:

                - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                  model).
                - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                  save directory.
                - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                  configuration JSON file named *config.json* is found in the directory.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            A state dictionary to use instead of a state dictionary loaded from saved weights file.

            This option can be used if you want to create a model from a pretrained configuration but load your own
            weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
            [`~PreTrainedModel.from_pretrained`] is not a simpler option.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        from_tf (`bool`, *optional*, defaults to `False`):
            Load the model weights from a TensorFlow checkpoint save file (see docstring of
            `pretrained_model_name_or_path` argument).
        from_flax (`bool`, *optional*, defaults to `False`):
            Load the model weights from a Flax checkpoint save file (see docstring of
            `pretrained_model_name_or_path` argument).
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
            as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
            checkpoint with 3 labels).
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received files. Will attempt to resume the download if such a
            file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        output_loading_info(`bool`, *optional*, defaults to `False`):
            Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
        local_files_only(`bool`, *optional*, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
            the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.

            <Tip>

            To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

            </Tip>

        mirror (`str`, *optional*):
            Mirror source to accelerate downloads in China. If you are from China and have an accessibility
            problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
            Please refer to the mirror site for more information.
        _fast_init(`bool`, *optional*, defaults to `True`):
            Whether or not to disable fast initialization.

            <Tip warning={true}>

            One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
            4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
            [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

            </Tip>

        > Parameters for big model inference

        low_cpu_mem_usage(`bool`, *optional*):
            Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
            This is an experimental feature and a subject to change at any moment.
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Override the default `torch.dtype` and load the model under a specific `dtype`. The different options
            are:

            1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
              `dtype`, ignoring the model's `config.torch_dtype` if one exists. If not specified
              - the model will get loaded in `torch.float` (fp32).

            2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be
              attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
              the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
              using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
              the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

            <Tip>

            For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
            reach out to the authors and ask them to add this information to the model's card and to insert the
            `torch_dtype` entry in `config.json` on the hub.

            </Tip>

        device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each
            parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
            same device.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
            more information about each option see [designing a device
            map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
            GPU and the available CPU RAM if unset.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
            RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
            `True` when there is some disk offload.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            If `True`, will convert the loaded model into mixed-8bit quantized model. To use this feature please
            install `bitsandbytes` compiled with your CUDA version by running `pip install -i
            https://test.pypi.org/simple/ bitsandbytes-cudaXXX` where XXX is your CUDA version (e.g. 11.6 = 116).
            Make also sure that you have enough GPU RAM to store half of the model size since the 8bit modules are
            not compiled and adapted for CPUs.
        quantization_config (`Dict`, *optional*):
            A dictionary of configuration parameters for the `bitsandbytes` library and loading the model using
            advanced features such as offloading in fp32 on CPU or on disk.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        variant (`str`, *optional*):
            If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
            ignored when using `from_tf` or `from_flax`.

        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
            `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
            automatically loaded:

                - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                  underlying model's `__init__` method (we assume all relevant updates to the configuration have
                  already been done)
                - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                  initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                  corresponds to a configuration attribute will be used to override said attribute with the
                  supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                  will be passed to the underlying model's `__init__` function.

    <Tip>

    Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
    use this method in a firewalled environment.

    </Tip>

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Download model and configuration from huggingface.co and cache.
    >>> model = BertModel.from_pretrained("bert-base-uncased")
    >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
    >>> model = BertModel.from_pretrained("./test/saved_model/")
    >>> # Update configuration during loading.
    >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    >>> assert model.config.output_attentions == True
    >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
    >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
    >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
    >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
    >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
    ```

    * `low_cpu_mem_usage` algorithm:

    This is an experimental function that loads the model using ~1x model size CPU memory

    Here is how it works:

    1. save which state_dict keys we have
    2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
    3. after the model has been instantiated switch to the meta device all params/buffers that
    are going to be replaced from the loaded state_dict
    4. load state_dict 2nd time
    5. replace the params/buffers from the state_dict

    Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

    """
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    from_flax = kwargs.pop("from_flax", False)
    ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    _ = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    _fast_init = kwargs.pop("_fast_init", True)
    torch_dtype = kwargs.pop("torch_dtype", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
    device_map = kwargs.pop("device_map", None)
    max_memory = kwargs.pop("max_memory", None)
    offload_folder = kwargs.pop("offload_folder", None)
    offload_state_dict = kwargs.pop("offload_state_dict", False)
    load_in_8bit = kwargs.pop("load_in_8bit", False)
    quantization_config = kwargs.pop("quantization_config", None)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    variant = kwargs.pop("variant", None)
    use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

    if is_bitsandbytes_available():
        is_8bit_serializable = version.parse(importlib_metadata.version("bitsandbytes")) > version.parse("0.37.2")
    else:
        is_8bit_serializable = False

    if trust_remote_code is True:
        logger.warning(
            "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
            " ignored."
        )
    if device_map is not None:
        if low_cpu_mem_usage is None:
            low_cpu_mem_usage = True
        elif not low_cpu_mem_usage:
            raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

    if low_cpu_mem_usage:
        if device_map is not None:
            # The max memory utils require PyTorch >= 1.10 to have torch.cuda.mem_get_info.
            require_version_core("torch>=1.10")

        if is_deepspeed_zero3_enabled():
            raise ValueError(
                "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
            )
        elif not is_accelerate_available():
            raise ImportError(
                "Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`"
            )

    if quantization_config is None:
        quantization_config, kwargs = BitsAndBytesConfig.from_dict(
            config_dict={"load_in_8bit": load_in_8bit}, return_unused_kwargs=True, **kwargs
        )
    elif quantization_config is not None:
        load_in_8bit = quantization_config.load_in_8bit

        quantization_config_kwargs = {
            k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters
        }

        if len(quantization_config_kwargs) > 0:
            raise ValueError(
                "You can't pass `load_in_8bit` or any other `BitsAndBytesConfig` argument as a kwarg when passing "
                "`quantization_config` argument at the same time."
            )

        # in the case a user loads an 8bit model from the Hub and assigns a new quantization_config
        if device_map is None:
            device_map = "auto"
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True

    if load_in_8bit:
        if not (is_accelerate_available() and is_bitsandbytes_available()):
            raise ImportError(
                "Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of"
                " bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or"
                " pip install bitsandbytes` "
            )
        if torch_dtype != torch.float16:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.warning(
                f"Overriding torch_dtype={torch_dtype} with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in mixed int8. "
                "Either pass torch_dtype=torch.float16 or don't pass this argument at all to remove this warning."
            )
            torch_dtype = torch.float16

        if device_map is None:
            raise ValueError(
                "A device map needs to be passed to run convert models into mixed-int8 format. Please run"
                "`.from_pretrained` with `device_map='auto'`"
            )
        if from_tf or from_flax:
            raise ValueError(
                "Converting into mixed 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

    from_pt = not (from_tf | from_flax)

    user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    if is_8bit_serializable and quantization_config is not None and load_in_8bit:
        if hasattr(config, "quantization_config"):
            logger.warning(
                "You passed `quantization_config` to `from_pretrained` but the model you're loading already has a"
                " `quantization_config` attribute. The `quantization_config` attribute will be overwritten with the"
                " one you passed to `from_pretrained`."
            )
        config.quantization_config = quantization_config
    elif is_8bit_serializable and not load_in_8bit and hasattr(config, "quantization_config"):
        quantization_config = config.quantization_config
        if isinstance(quantization_config, dict):
            quantization_config = BitsAndBytesConfig.from_dict(quantization_config, return_unused_kwargs=False)
        elif isinstance(quantization_config, BitsAndBytesConfig):
            pass
        else:
            raise ValueError(
                f"Invalid type for `quantization_config`: {type(quantization_config)}. Should be a `dict` or a"
                " `BitsAndBytesConfig` instance."
            )

        load_in_8bit = quantization_config.load_in_8bit

        if load_in_8bit:
            torch_dtype = torch.float16

            if device_map is None:
                device_map = "auto"

            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
    elif not is_8bit_serializable and not load_in_8bit and hasattr(config, "quantization_config"):
        logger.warning(
            "Detected the presence of a `quantization_config` attribute in the model's configuration but you don't have the correct"
            " `bitsandbytes` version to support int8 serialization. Please install the latest version of `bitsandbytes` with "
            " `pip install --upgrade bitsandbytes`."
        )

    if commit_hash is None:
        commit_hash = getattr(config, "_commit_hash", None)

    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    # index of the files.
    is_sharded = False
    sharded_metadata = None
    # Load model
    loading_info = None

    # Keep in fp32 modules
    keep_in_fp32_modules = None
    use_keep_in_fp32_modules = False

    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if from_tf and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            ):
                # Load from a TF 1.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            elif from_tf and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            ):
                # Load from a TF 2.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            elif from_flax and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            ):
                # Load from a Flax checkpoint in priority if from_flax
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                )
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            # At this stage we don't have a weight file so we will raise an error.
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                    " `from_tf=True` to load this model from those weights."
                )
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                    " to load this model from those weights."
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME},"
                    f" {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
            if not from_tf:
                raise ValueError(
                    f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                    "from_tf to True to load from this checkpoint."
                )
            archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
        else:
            # set correct filename
            if from_tf:
                filename = TF2_WEIGHTS_NAME
            elif from_flax:
                filename = FLAX_WEIGHTS_NAME
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            try:
                # Load from URL or cache if already cached
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "resume_download": resume_download,
                    "local_files_only": local_files_only,
                    "use_auth_token": use_auth_token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        raise EnvironmentError(
                            f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} and thus cannot be loaded with `safetensors`. Please make sure that the model has been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                        )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                if resolved_archive_file is None:
                    # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                    # message.
                    has_file_kwargs = {
                        "revision": revision,
                        "proxies": proxies,
                        "use_auth_token": use_auth_token,
                    }
                    if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                            " Use `from_tf=True` to load this model from those weights."
                        )
                    elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                            " `from_flax=True` to load this model from those weights."
                        )
                    elif variant is not None and has_file(
                        pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                    ):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                            f" {FLAX_WEIGHTS_NAME}."
                        )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                )

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
        else:
            logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
    else:
        resolved_archive_file = None

    # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    if is_sharded:
        # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )

    # load pt weights early so that we know which dtype to init the model under
    # pdb.set_trace()
    if from_pt:
        if not is_sharded and state_dict is None:
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_archive_file)

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
        #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
        # we also may have config.torch_dtype available, but we won't rely on it till v5
        dtype_orig = None

        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                        torch_dtype = config.torch_dtype
                        logger.info(f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            torch_dtype = get_state_dict_dtype(state_dict)
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file[0])
                            torch_dtype = get_state_dict_dtype(one_state_dict)
                            del one_state_dict  # free CPU memory
                        logger.info(
                            "Since the `torch_dtype` attribute can't be found in model's config object, "
                            "will use torch_dtype={torch_dtype} as derived from model's weights"
                        )
                else:
                    raise ValueError(
                        f'`torch_dtype` can be either `torch.dtype` or `"auto"`, but received {torch_dtype}'
                    )
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (
            (cls._keep_in_fp32_modules is not None) and is_accelerate_available() and torch_dtype == torch.float16
        )
        if (
            (cls._keep_in_fp32_modules is not None)
            and not is_accelerate_available()
            and torch_dtype == torch.float16
        ):
            logger.warning(
                "For stability purposes, it is recommended to have accelerate installed when using this model in"
                " torch.float16, please install it with `pip install accelerate`"
            )

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_state_dict_keys = list(state_dict.keys())
        if low_cpu_mem_usage or use_keep_in_fp32_modules:
            state_dict = None

    config.name_or_path = pretrained_model_name_or_path

    # Instantiate model.
    init_contexts = [no_init_weights(_enable=_fast_init)]

    if is_deepspeed_zero3_enabled():
        import deepspeed

        logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
        init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())] + init_contexts
    elif load_in_8bit or low_cpu_mem_usage:
        init_contexts.append(init_empty_weights())

    with ContextManagers(init_contexts):
        model = cls(config, *model_args, **model_kwargs)

    # Check first if we are `from_pt`
    if use_keep_in_fp32_modules:
        low_cpu_mem_usage = True
        keep_in_fp32_modules = model._keep_in_fp32_modules
    else:
        keep_in_fp32_modules = []

    if load_in_8bit:
        from transformers.utils.bitsandbytes import get_keys_to_not_convert, replace_8bit_linear

        load_in_8bit_skip_modules = quantization_config.llm_int8_skip_modules
        load_in_8bit_threshold = quantization_config.llm_int8_threshold
        load_in_8bit_fp32_cpu_offload = quantization_config.llm_int8_enable_fp32_cpu_offload

        logger.info("Detected 8-bit loading: activating 8-bit loading for this model")

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if load_in_8bit_skip_modules is None:
            modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            modules_to_not_convert = load_in_8bit_skip_modules

        if not isinstance(modules_to_not_convert, list):
            modules_to_not_convert = [modules_to_not_convert]

        modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend the modules to not convert to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )

            modules_to_not_convert.extend(keys_on_cpu)

        model = replace_8bit_linear(
            model, threshold=load_in_8bit_threshold, modules_to_not_convert=modules_to_not_convert
        )

        # training in 8-bit is only available in 0.37.0+
        model._is_int8_training_enabled = version.parse(
            importlib_metadata.version("bitsandbytes")
        ) >= version.parse("0.37.0")

        model.config.quantization_config = quantization_config
        model.is_8bit_serializable = is_8bit_serializable

    if isinstance(device_map, str):
        special_dtypes = {}
        if load_in_8bit:
            special_dtypes.update(
                {
                    name: torch_dtype
                    for name, _ in model.named_parameters()
                    if any(m in name for m in modules_to_not_convert)
                }
            )

        special_dtypes.update(
            {
                name: torch.float32
                for name, _ in model.named_parameters()
                if any(m in name for m in keep_in_fp32_modules)
            }
        )

        if model._no_split_modules is None:
            raise ValueError(f"{model.__class__.__name__} does not support `device_map='{device_map}'` yet.")
        no_split_modules = model._no_split_modules
        if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )
        elif device_map in ["balanced", "balanced_low_0"] and get_balanced_memory is None:
            raise ValueError(f"`device_map={device_map}` requires a source install of Accelerate.")

        kwargs = {"no_split_module_classes": no_split_modules}
        if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
            kwargs["special_dtypes"] = special_dtypes
        elif len(special_dtypes) > 0:
            logger.warn(
                "This model has some weights that should be kept in higher precision, you need to upgrade "
                "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
            )
        if device_map != "sequential" and get_balanced_memory is not None:
            max_memory = get_balanced_memory(
                model,
                dtype=torch_dtype if not load_in_8bit else torch.int8,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **kwargs,
            )
        kwargs["max_memory"] = max_memory
        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(model, dtype=torch_dtype if not load_in_8bit else torch.int8, **kwargs)

        if load_in_8bit:
            # The LM head / tied weights or any last module can stay on disk / CPU
            device_map_without_lm_head = {
                key: device_map[key] for key in device_map.keys() if key not in modules_to_not_convert
            }
            if "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
                raise ValueError(
                    """
                    Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit
                    the quantized model. If you want to dispatch the model on the CPU or the disk while keeping
                    these modules in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom
                    `device_map` to `from_pretrained`. Check
                    https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
                    for more details.
                    """
                )
            del device_map_without_lm_head

    if from_tf:
        if resolved_archive_file.endswith(".index"):
            # Load from a TensorFlow 1.X checkpoint - provided by original authors
            model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
        else:
            # Load from our TensorFlow 2.0 checkpoints
            try:
                from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                model, loading_info = load_tf2_checkpoint_in_pytorch_model(
                    model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True
                )
            except ImportError:
                logger.error(
                    "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
                    " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
                    " instructions."
                )
                raise
    elif from_flax:
        try:
            from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

            model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
        except ImportError:
            logger.error(
                "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
                " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
                " installation instructions."
            )
            raise
    elif from_pt:
        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            state_dict,
            loaded_state_dict_keys,  # XXX: rename?
            resolved_archive_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            keep_in_fp32_modules=keep_in_fp32_modules,
        )

    model.is_loaded_in_8bit = load_in_8bit

    # make sure token embedding weights are still tied if needed
    model.tie_weights()

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    # If it is a model with generation capabilities, attempt to load the generation config
    if model.can_generate():
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        except (OSError, TypeError):
            logger.info(
                "Generation config file not found, using a generation config created from the model config."
            )
            pass

    # Dispatch model with hooks on all devices if necessary
    if device_map is not None:
        dispatch_model(model, device_map=device_map, offload_dir=offload_folder, offload_index=offload_index)

    if output_loading_info:
        if loading_info is None:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
        return model, loading_info

    return model


def replace_llama_with_thought():
    # transformers.models.llama.modeling_llama.LlamaModel.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward2
    transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaForCausalLM._load_pretrained_model = _load_pretrained_model2
    transformers.models.llama.modeling_llama.LlamaForCausalLM.from_pretrained = from_pretrained2
