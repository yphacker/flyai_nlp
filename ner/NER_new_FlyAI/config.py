# coding=utf-8
# author=yphacker

import os
import sys
import json

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')
# bert路径
BERT_PATH = os.path.join(sys.path[0], 'bert', )

src_vocab_file = os.path.join(DATA_PATH, 'words.dict')
word_embedding_file = os.path.join(DATA_PATH, 'embedding.json')

with open(os.path.join(DATA_PATH, 'words.dict'), 'r') as vocab_file:
    vocab_size = len(json.load(vocab_file))

src_unknown_id = vocab_size
src_padding = vocab_size + 1
label_dic = ['B-LAW', 'B-ROLE', 'B-TIME', 'I-LOC', 'I-LAW', 'B-PER', 'I-PER', 'B-ORG', 'I-ROLE', 'I-CRIME', 'B-CRIME',
             'I-ORG', 'B-LOC', 'I-TIME', 'O', 'padding']
num_labels = len(label_dic)

max_seq_length = 100
embeddings_size = 200
batch_size = 32

print_per_batch = 10
improvement_step = print_per_batch * 10

lstm_size = 200
keep_prob = 0.5
learning_rate = 0.001
