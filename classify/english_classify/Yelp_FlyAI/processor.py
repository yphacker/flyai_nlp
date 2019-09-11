# -*- coding: utf-8 -*
# import numpy
# from flyai.processor.base import Base
#
#
# class Processor(Base):
#
#     def input_x(self, text):
#         '''
#         参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
#         '''
#         return text
#
#     def input_y(self, stars):
#         '''
#         参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
#         '''
#         one_hot_label = numpy.zeros([5])  ##生成全0矩阵
#         one_hot_label[stars - 1] = 1  ##相应标签位置置
#         return one_hot_label
#
#     def output_y(self, data):
#         '''
#         验证时使用，把模型输出的y转为对应的结果
#         '''
#         return numpy.argmax(data)

import os
from flyai.processor.base import Base
import config
import bert.tokenization as tokenization
from bert.run_classifier import convert_single_example_simple


class Processor(Base):
    def __init__(self):
        self.token = None

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''

        if self.token is None:
            # bert_vocab_file = os.path.join(config.DATA_PATH, "model", "uncased_L-24_H-1024_A-16", 'vocab.txt')
            bert_vocab_file = os.path.join(config.DATA_PATH, "model", "uncased_L-12_H-768_A-12", 'vocab.txt')
            self.token = tokenization.FullTokenizer(vocab_file=bert_vocab_file)
        word_ids, word_mask, word_segment_ids = convert_single_example_simple(max_seq_length=config.max_seq_length,
                                                                              tokenizer=self.token, text_a=text)
        return word_ids, word_mask, word_segment_ids

    def input_y(self, stars):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return stars

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return data[0]
