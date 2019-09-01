# -*- coding: utf-8 -*- 

import os
import json
from path import DATA_PATH
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_dict():
    dict_path = os.path.join(DATA_PATH, 'words.dict')
    with open(dict_path, 'r', encoding='utf-8') as fin:
        word_dict = json.load(fin)
    return word_dict


def word2id(sent, word_dict, max_seq_len=34):
    if len(sent) == 0 or len(word_dict) == 0:
        print('[ERROR] word2id failed! | The params {} and {}'.format(sent, word_dict))
        return None

    sent_list = sent.strip().split()
    sent_ids = list()
    for item in sent_list:
        if item in word_dict:
            sent_ids.append(word_dict[item])
        else:
            sent_ids.append(word_dict['_unk_'])
    if len(sent_ids) < max_seq_len:
        sent_ids = sent_ids + [word_dict['_pad_'] for _ in range(max_seq_len - len(sent_ids))]
    else:
        sent_ids = sent_ids[:max_seq_len]

    return sent_ids


if __name__ == "__main__":

    exit(1)