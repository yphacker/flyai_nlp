# -*- coding: utf-8 -*

import os
import json
from flyai.processor.base import Base
import config
import bert.tokenization as tokenization
from bert.run_classifier import convert_single_example_simple


# class Processor(Base):
#     def __init__(self):
#         self.token = None
#         self.label_dic = config.label_dic
#         with open(config.src_vocab_file, 'r') as fw:
#             self.words_dic = json.load(fw)
#
#     def input_x(self, source):
#         '''
#         参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
#         '''
#         sen2id = []
#         source = source.split()
#         for s in source:
#             if s in self.words_dic:
#                 sen2id.append(self.words_dic[s])
#             else:
#                 sen2id.append(config.src_unknown_id)
#         return sen2id
#
#     def input_y(self, target):
#         '''
#         参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
#         '''
#         label2id = []
#         target = target.split()
#         for t in target:
#             label2id.append(self.label_dic.index(t))
#         return label2id
#
#     def output_y(self, index):
#         '''
#         验证时使用，把模型输出的y转为对应的结果
#         '''
#         label = []
#         for i in index:
#             if i != config.num_labels - 1:
#                 label.append(config.label_dic[i])
#             else:
#                 break
#         return label

class Processor(Base):
    def __init__(self):
        self.token = None
        # self.label_dic = config.label_dic
        # with open(config.src_vocab_file, 'r') as fw:
        #     self.words_dic = json.load(fw)

    def input_x(self, source):
        '''
            参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        if self.token is None:
            bert_vocab_file = os.path.join(config.DATA_PATH, "model", "uncased_L-24_H-1024_A-16", 'vocab.txt')
            self.token = tokenization.FullTokenizer(vocab_file=bert_vocab_file)
        word_ids, word_mask, word_segment_ids = convert_single_example_simple(max_seq_length=config.max_seq_length,
                                                                              tokenizer=self.token, text_a=source)
        return word_ids, word_mask, word_segment_ids

    def input_y(self, target):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        label2id = []
        target = target.split()
        for t in target:
            label2id.append(self.label_dic.index(t))
        return label2id

    def output_y(self, index):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        label = []
        for i in index:
            if i != config.num_labels - 1:
                label.append(config.label_dic[i])
            else:
                break
        return label
