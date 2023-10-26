import json
import re
from tqdm import tqdm


src_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_train.json'
ref_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data_old/toolllama_G123_dfs_train.json'
tar_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_train_light.json'


# raw_data = json.load(open(src_file, "r"))
# ref_data = json.load(open(ref_file, 'r'))
#
# ref_dict = {}
# for t in tqdm(ref_data):
#     id = t['id']
#     step = re.findall('Step .*:', id)[0]
#     query = id.split(step)[1].strip()
#     ref_dict[query] = 1
#
#
# out_data = []
# out_dict = {}
# for item in tqdm(raw_data):
#     id = item['id']
#     step = re.findall('Step .*:', id)[0]
#     query = id.split(step)[1].strip()
#     if query in ref_dict:
#         out_data.append(item)
#         out_dict[query] = 1
#     else:
#         # print(query)
#         pass
#
# with open(tar_file, 'w') as f:
#     json.dump(out_data, f, indent=2)


'''get the data without thought'''
# tar_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_train_light.json'
# tar_file2 = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_train_light_wo_thought.json'

tar_file = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_eval.json'
tar_file2 = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/toolllama_G123_dfs_eval_wo_thought.json'

out_data = json.load(open(tar_file, "r"))
out_data2 = []

for item in out_data:
    convs = item['conversations']
    convs_new = []
    for c in convs:
        if c['from'] == 'assistant':
            value = c['value']
            if value.startswith('\nThought'):
                parts = value.split('\nAction')
                value_new = '\nThought:'
                for t in parts[1:]:
                    value_new += '\nAction' + t
                c['value'] = value_new
        convs_new.append(c)
    item['conversations'] = convs_new
    out_data2.append(item)

with open(tar_file2, 'w') as f:
    json.dump(out_data2, f, indent=2)




# source_dict = {}
# num_double = 0
