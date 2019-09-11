# coding=utf-8
# author=yphacker

import os
import sys

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')
# bert路径
BERT_PATH = os.path.join(sys.path[0], 'bert', )

print_per_batch = 10
improvement_step = print_per_batch * 10

# max_seq_length = 128
# num_labels = 3  # 类别数
# learning_rate = 5e-5
# grad_clip = 5.0

max_seq_len = 200
numclass = 2

embedding_dim = 64  # 嵌入层大小
dnn_dim = 128  # Dense层大小
num_filters = 64  # 卷积核数目
kernel_size = 5  # 卷积核尺寸
learning_rate = 1e-3  # 学习率
