# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:43:45 2018

Changed on 2019.1.3 by @guandan

@author: Shelton
"""
import pandas
import json
import re
import os


def make_embedding_for_json(save_path, seg_list, embedding_path=os.path.join('/data', 'embedding','glove.840B.300d.txt')):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    embeddin_map = {}
    f = open(embedding_path, encoding='utf8', mode='r')
    embedding_index = json.loads(f.read())
    for seg in seg_list:
        if seg not in embeddin_map and seg in embedding_index:
            embeddin_map[seg] = embedding_index[seg]
    f = open(os.path.join(save_path, "embedding.json"), encoding='utf-8', mode='w')
    f.write(json.dumps(embeddin_map))

dict_path = './data/word.dict'
with open(dict_path, encoding='utf-8') as fin:
    word_dict = json.load(fin)
    seg_list = list(word_dict.keys())
make_embedding_for_json("/data/caption", seg_list)