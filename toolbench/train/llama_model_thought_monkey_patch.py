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
from transformers.modeling_utils import \
    (set_module_tensor_to_device, set_initialized_submodules,
     _load_state_dict_into_model, expand_device_map, logging,
     get_disk_only_shard_files, load_state_dict, _load_state_dict_into_meta_model,
     save_offload_index, load_offloaded_weights, logger)
import gc
import os
import re
import shutil
import tempfile
import pdb

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


@classmethod
def _load_pretrained_model(
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
    pdb.set_trace()

    print('state_dict')
    print(state_dict)
    print('pretrained_model_name_or_path')
    print(pretrained_model_name_or_path)
    print('loaded keys')
    print(loaded_keys)

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
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    add_prefix_to_model = has_prefix_module and not expects_prefix_module

    if remove_prefix_from_model:
        # _prefix = f"{prefix}."
        # expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
        # expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]

        '''model_teacher'''
        expected_keys_not_prefixed = []
        expected_keys_new = []
        prefix2 = 'model_teacher'
        _prefix = f"{prefix}."
        _prefix2 = f"{prefix2}."
        for s in expected_keys:
            if s.startswith(_prefix):
                expected_keys_new.append(s[len(_prefix):])
            elif s.startswith(_prefix2):
                expected_keys_new.append(s[len(_prefix2):])
            else:
                expected_keys.append(s)
                expected_keys_not_prefixed.append(s)
        expected_keys = expected_keys_new

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
    print(len(cls.base_model_prefix))
    print(hasattr(model, cls.base_model_prefix))
    print(has_prefix_module)

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

        '''model teacher'''
        print('load teacher model')
        error_msgs += _load_state_dict_into_model(model.model_teacher, state_dict, start_prefix)

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
                '''model teacher'''
                print('load student')
                print('start_prefix', start_prefix)
                print(type(model))
                print(type(model_to_load))
                print(list(state_dict.keys())[0])

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

                '''model teacher'''
                print('load teacher')
                print('start_prefix', start_prefix)
                print(type(model))
                print(list(state_dict.keys())[0])

                new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                    model.model_teacher,
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

            else:
                error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
                '''model teacher'''
                print('load teacher')
                error_msgs += _load_state_dict_into_model(model.model_teacher, state_dict, start_prefix)

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


def replace_llama_with_thought():
    # transformers.models.llama.modeling_llama.LlamaModel.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward2
    transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaForCausalLM._load_pretrained_model = _load_pretrained_model
