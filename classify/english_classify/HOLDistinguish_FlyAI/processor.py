# -*- coding: utf-8 -*

import os
import re
from flyai.processor.base import Base
import bert.tokenization as tokenization
from bert.run_classifier import convert_single_example_simple
import config
import html
from string import punctuation


# class Processor(Base):
#     def __init__(self):
#         super(Processor, self).__init__()
#         self.word_dict = load_dict()
#
#     def input_x(self, text):
#         '''
#         参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
#         和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
#         该方法字段与app.yaml中的input:->columns:对应
#         '''
#
#         sent_ids = word2id(text, self.word_dict, MAX_LEN)
#
#         return sent_ids
#
#     def input_y(self, label):
#         '''
#         参 数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
#         和dataset.next_validation_batch()多次调用。
#         该方法字段与app.yaml中的output:->columns:对应
#         '''
#         # 0 - hate speech
#         # 1 - offensive language
#         # 2 - neither
#         return int(label)
#
#     def output_y(self, data):
#         '''
#         输出的结果，会被dataset.to_categorys(data)调用
#         '''
#
#         return data[0]

class Processor(Base):
    def __init__(self):
        self.token = None

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        if self.token is None:
            bert_vocab_file = os.path.join(config.DATA_PATH, "model", "uncased_L-24_H-1024_A-16", 'vocab.txt')
            self.token = tokenization.FullTokenizer(vocab_file=bert_vocab_file)
        # pattern = "[!]+"
        # text = re.sub(pattern, '', text)
        text = html.unescape(text)

        # pattern = re.compile(r'[0-9]|[%s]+' % punctuation)
        # tmp = re.sub(pattern, '', text)
        # tmp = tmp.split(' ')
        # new_tmp = []
        # for word in tmp:
        #     if word != '':
        #         new_tmp.append(word)
        # text = ' '.join(new_tmp)

        word_ids, word_mask, word_segment_ids = \
            convert_single_example_simple(max_seq_length=config.max_seq_length, tokenizer=self.token, text_a=text)

        return word_ids, word_mask, word_segment_ids

    def input_y(self, label):
        '''
        参 数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。
        该方法字段与app.yaml中的output:->columns:对应
        '''
        # 0 - hate speech
        # 1 - offensive language
        # 2 - neither
        return int(label)

    def output_y(self, data):
        '''
        输出的结果，会被dataset.to_categorys(data)调用
        '''
        return data[0]
