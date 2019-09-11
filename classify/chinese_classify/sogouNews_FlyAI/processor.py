# -*- coding: utf-8 -*
from flyai.processor.download import check_download
from flyai.processor.base import Base
from path import DATA_PATH  # 导入输入数据的地址
import os
import jieba
import json
import numpy as np
import create_dict

MAX_LEN = 50

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        self.word_dict, self.word_dict_res = create_dict.load_dict()
        self.label_dict, self.label_dict_res = create_dict.load_label_dict()
        # self.seq_len = 50

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        terms = jieba.cut(text, cut_all=False)
        # terms = list(terms)
        truncate_terms = []
        for term in terms:
            truncate_terms.append(term)
            if len(truncate_terms)>=MAX_LEN:
                break
        index_list = [self.word_dict[term] if term in self.word_dict
                       else create_dict._UNK_ for term in truncate_terms ]
        if len(index_list)<MAX_LEN:
            index_list = index_list+[create_dict._PAD_]*(MAX_LEN-len(index_list))
        return index_list


    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        y = np.zeros(len(self.label_dict))
        if label in self.label_dict:
            label_index = self.label_dict[label]
            y[label_index] = 1
        return y

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        out_y = np.argmax(data)
        if out_y in self.label_dict_res:
            out_y = self.label_dict_res[out_y]
        else:
            out_y = '未知标签'
        return out_y
