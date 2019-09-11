# -*- coding: utf-8 -*
import pandas
import json
import re
import os
from path import DATA_PATH
import sys
import jieba
special_chars = ['_PAD_', '_EOS_', '_SOS_', '_UNK_']#'_START_'
_PAD_ = 0
_EOS_ = 1
_UNK_ = 2
_SOS_ = 3

#_START_ = 3
def create(file_dir, DICT_PATH, LABEL_PATH):
    print('save to', DICT_PATH, LABEL_PATH)
    word_dict = dict()
    for i, word in enumerate(special_chars):
        word_dict[word] = i
    label_dict = dict()
    filenames = os.listdir(file_dir)#os.walk(file_dir, ):
    for filename in filenames:
        if filename.endswith('.csv'):
            print(filename)
            f = pandas.read_csv(os.path.join(file_dir, filename), usecols=['text', 'label'])
            labels = f.values[:, 1]
            labels = labels.astype('str')
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = len(label_dict)

            data = f.values[:,0]
            data = data.astype('str')
            for sentence in data:
                # keep_len = 1000
                # if len(sentence)>keep_len:
                #     sentence = sentence[:keep_len]
                terms = jieba.cut(sentence, cut_all=False)
                #sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", sentence)
                for c in terms:
                    if c not in word_dict:
                        word_dict[c] = len(word_dict)
    with open(os.path.join(DICT_PATH), 'w', encoding='utf-8') as fout:
        json.dump(word_dict, fout)
    with open(os.path.join(LABEL_PATH), 'w', encoding='utf-8') as fout:
        json.dump(label_dict, fout)
    print('build dict done.')

def load_dict():
    char_dict_re = dict()
    dict_path = os.path.join(DATA_PATH, 'word.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re

def load_label_dict():
    char_dict_re = dict()
    dict_path = os.path.join(DATA_PATH, 'label.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re

if __name__ == '__main__':
    create('./data/', DICT_PATH=os.path.join(sys.path[0], 'data', 'word.dict'),
           LABEL_PATH=os.path.join(sys.path[0], 'data', 'label.dict'))
