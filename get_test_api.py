import json
from tqdm import tqdm
import os
from os.path import join

def get_test_api(file_path, id_path):
    querys = json.load(open(file_path, "r"))

    ids = json.load(open(id_path, 'r'))
    ids = list(ids.keys())
    ids = [int(id) for id in ids]

    filtered_querys = []
    for q in querys:
        if q['query_id'] in ids:
            filtered_querys.append(q)
    querys = filtered_querys

    api_list = []
    for q in tqdm(querys):
        for api in q['api_list']:
            tool_name = api['tool_name']
            if tool_name not in api_list:
                api_list.append(tool_name)
                print(tool_name)
        for api in q['relevant APIs']:
            tool_name = api[0]
            if tool_name not in api_list:
                api_list.append(tool_name)
                print(tool_name)

    return api_list

if __name__ == '__main__':
    file_dir = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/instruction'
    id_dir = '/home/SENSETIME/zengwang/myprojects/llm/ToolBench/data/test_query_ids'
    api_list = []
    for id_file in os.listdir(id_dir):
        id_path = join(id_dir, id_file)
        if 'G1' in id_file:
            file_path = join(file_dir, 'G1_query.json')
        if 'G2' in id_file:
            file_path = join(file_dir, 'G2_query.json')
        if 'G3' in id_file:
            file_path = join(file_dir, 'G3_query.json')

        print(file_path, id_path)
        api_list += get_test_api(file_path, id_path)
        api_list = list(sorted(list(set(api_list))))
        print(len(api_list))
    t = 0


with open('test_api_list.txt', 'w') as f:
    for apiname in api_list:
        f.write(apiname.lower()+'\n')
