import json
import re
import os
import json
import time
import requests
from tqdm import tqdm
from termcolor import colored
import random
# from toolbench.inference.Algorithms.single_chain import single_chain


def print_tree(node):
    color_converter = {
        "Thought": "red",
        "Action": "white",
        "Action Input": "cyan",
        "Final Answer": "green",
        "Reflection": "blue",
        "Observation": "yellow"
    }
    node_type = node['node_type']
    description = node['description']
    observation = node['observation'] if 'observation' in node else ""

    print(colored(f"{node_type}: {description}", color=color_converter[node_type]))

    if observation != "":
        if len(observation) < 1536:
            print(colored(f"Observation: {observation}", color=color_converter["Observation"]))
        else:
            print(colored(f"Observation: {observation[:1536]}......(len={len(observation)})", color=color_converter["Observation"]))

    for child in node['children']:
        print_tree(child)


role_to_color = {
    "system": "blue",
    "user": "green",
    "assistant": "red",
    "function": "yellow",
}

# res_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/work_dirs/answer/thought/baseline/G1_category/1185_DFS_woFilter_w2.json'
res_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/work_dirs/answer/thought/split/G1_category/1185_DFS_woFilter_w2.json'
data = json.load(open(res_file, 'r'))

q = data["answer_generation"]['query']
tree = data['tree']['tree']
print(colored(q, 'green'))
print_tree(tree)


#
# q = data["answer_generation"]['query']
# answers = data["answer_generation"]['train_messages']
#
# print(colored(f'user: {q}\n', 'green'))
# for ans_list in answers:
#     for message in ans_list:
#         if message['role'] in ['system', 'user']:
#             pass
#             continue
#
#         print_obj = f"{message['role']}: {message['content']} "
#         if "function_call" in message.keys():
#             print_obj = print_obj + f"\nfunction_call: {message['function_call']}"
#         print_obj += "\n"
#
#         print(
#             colored(
#                 print_obj,
#                 role_to_color[message["role"]],
#             )
#         )
#     print(colored('-'*120, 'white'))
#



t = 0