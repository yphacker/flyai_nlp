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

print_per_batch = 10
improvement_step = print_per_batch * 10
learning_rate = 0.001
