# -*- coding: utf-8 -*- 

import re
import os
import json
import config


def data_clean(sent):
    """保留句子中的英文，并转换为小写"""
    string_list = re.sub("[^A-Za-z0-9\'@]", " ", sent).strip().lower().split(' ')
    sent_list = list()
    for item in string_list:
        if len(item) == 0:
            continue
        if item[0] == "'" and len(item) == 1:
            continue
        if item[0] == "'":
            item = item[1:]
        elif item[-1] == "'":
            item = item[:-1]
        sent_list.append(item)

    return sent_list


def load_dict():
    dict_path = os.path.join(config.DATA_PATH, 'words.dict')
    with open(dict_path, encoding='utf-8') as fin:
        word_dict = json.load(fin)
    return word_dict


def word2id(sent, word_dict, max_seq_len=33):
    if len(sent) == 0 or len(word_dict) == 0:
        print('[ERROR] word2id failed! | The params {} and {}'.format(sent, word_dict))
        return None

    sent_list = data_clean(sent)
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
    pass
