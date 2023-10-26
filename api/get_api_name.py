import os
from os.path import join, exists

root_dir = '../data/toolenv/tools'

api_list = []
for dirname in os.listdir(root_dir):
    for apiname in os.listdir(join(root_dir, dirname)):
        api_list.append(apiname)

with open('data/api_list.txt', 'w') as f:
    for apiname in api_list:
        f.write(apiname+'\n')


