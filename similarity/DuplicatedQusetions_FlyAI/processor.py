# -*- coding: utf-8 -*

import os
import numpy as np
from flyai.processor.base import Base
import config
from bert import tokenization
from bert.run_classifier import convert_single_example_simple


# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
#
#
# class Processor(Base):
#     # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
#     def __init__(self):
#         super(Processor, self).__init__()
#         embedding_path = os.path.join(DATA_PATH, 'embedding.json')
#         with open(embedding_path, encoding='utf-8') as f:
#             self.vocab = json.loads(f.read())
#         self.max_sts_len = 30
#         self.embedding_len = 100
#
#     def input_x(self, question1, question2):
#         '''
#         参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
#         '''
#         # question1处理
#         question1 = str(question1)
#         word_list1 = []
#         question1 = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", question1)
#         question1 = question1.strip().split(' ')
#         for word in question1:
#             embedding_vector = self.vocab.get(word)
#             if embedding_vector is not None:
#                 if len(embedding_vector) == self.embedding_len:
#                     # 给出现在编码词典中的词汇编码
#                     embedding_vector = list(map(lambda x: float(x),
#                                                 embedding_vector))  ## convert element type from str to float in the list
#                     word_list1.append(embedding_vector)
#         if len(word_list1) >= self.max_sts_len:
#             word_list1 = word_list1[:self.max_sts_len]
#         else:
#             for i in range(len(word_list1), self.max_sts_len):
#                 word_list1.append([0 for j in range(self.embedding_len)])  ## 词向量维度为200
#         word_list1 = np.stack(word_list1)
#
#         ##question2处理
#         question2 = str(question2)
#         word_list2 = []
#         question2 = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", question2)
#         question2 = question2.strip().split(' ')
#         for word in question2:
#             embedding_vector = self.vocab.get(word)
#             if embedding_vector is not None:
#                 if len(embedding_vector) == self.embedding_len:
#                     # 给出现在编码词典中的词汇编码
#                     embedding_vector = list(map(lambda x: float(x),
#                                                 embedding_vector))  ## convert element type from str to float in the list
#                     word_list2.append(embedding_vector)
#         if len(word_list2) >= self.max_sts_len:
#             word_list2 = word_list2[:self.max_sts_len]
#         else:
#             for i in range(len(word_list2), self.max_sts_len):
#                 word_list2.append([0 for j in range(self.embedding_len)])  ## 词向量维度为200
#         word_list2 = np.stack(word_list2)
#         return word_list1, word_list2
#
#     # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
#     def input_y(self, labels):
#         '''
#         参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
#         '''
#         return labels
#
#     def output_y(self, data):
#         '''
#         验证时使用，把模型输出的y转为对应的结果
#         '''
#         labels = np.array(data)
#         labels = labels.astype(np.float32)
#         out_y = labels
#         return out_y


class Processor(Base):
    def __init__(self):
        self.token = None

    def input_x(self, question1, question2):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        if self.token is None:
            bert_vocab_file = os.path.join(config.DATA_PATH, "model", "uncased_L-12_H-768_A-12", 'vocab.txt')
            self.token = tokenization.FullTokenizer(vocab_file=bert_vocab_file)
        question1 = question1.lower() if question1 is not np.nan else ''
        question2 = question2.lower() if question2 is not np.nan else ''
        word_ids, word_mask, word_segment_ids = convert_single_example_simple(max_seq_length=config.max_seq_length,
                                                                              tokenizer=self.token,
                                                                              text_a=question1, text_b=question2)
        return word_ids, word_mask, word_segment_ids

    def input_y(self, labels):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return labels

    def output_y(self, labels):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return labels
