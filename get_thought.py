from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from toolbench.tool_conversation import SeparatorStyle
from toolbench.model.model_adapter import get_conversation_template
import re
from tqdm import tqdm


def is_contained(conv1, conv2):
    return conv1 == conv2[:len(conv1)]


src_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_train.json'
# tar_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_train_filtered.json'


raw_data = json.load(open(src_file, "r"))
source_dict = {}
num_double = 0

# Apply prompt templates
sources = [example["conversations"] for example in raw_data]

template = "tool-llama-single-round"
conv = get_conversation_template(template)
if template == "tool-llama":
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
    roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

num_assit = 0
num_thought = 0
num_give_up = 0
num_answer = 0
thoughts = []
conversations = []
for i, source in enumerate(sources):
    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        conv.append_message(role, sentence["value"])

        # find out how many thought
        if role == 'Assistant':
            num_assit += 1
            s = sentence["value"]
            if not s.startswith('\nThought: \n'):
                num_thought += 1
                thought = s.split('Thought:')[1].split('Action')[0]
                # print(thought)
                thoughts.append(thought)

            if 'give_up' in s:
                num_give_up += 1
                # print(s)
            if 'give_answer' in s.lower():
                num_answer += 1
                # print(s)

            print('thought, answer, give up / all: {}, {}, {} / {}'.format(num_thought, num_answer, num_give_up, num_assit))

    conversations.append(conv.get_prompt())




output_file = 'data/train_thoughts.txt'
# output_file = 'data_0830/train_thoughts.txt'
with open(output_file, 'w') as f:
    for thought in thoughts:
        f.write('='*30+'\n')
        f.write(thought+'\n')
        # f.write('='*30+'\n')

output_file = 'data/train_conversation.txt'
# output_file = 'data_0830/train_conversation.txt'
with open(output_file, 'w') as f:
    for conver in conversations:
        # os.system('clear')
        f.write('=================================================================================')
        f.write('\n')
        f.write(conver)
        f.write('\n')

print('finish')