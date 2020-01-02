# coding=utf-8
# author=yphacker

# rnn神经元单元的状态数
hidden_size = 256
# rnn神经元单元类型，可以为lstm或gru
cell_type = 'lstm'
# 编码器和解码器的层数
layer_size = 4
# 词嵌入的维度
embedding_dim = 300
# 编码器和解码器是否共用词嵌入
# share_embedding = True
share_embedding = False
# 解码允许的最大步数
max_decode_step = 80
# 梯度裁剪的阈值
max_gradient_norm = 3.0
# 学习率初始值
learning_rate = 0.001
decay_step = 100000
# 学习率允许的最小值
min_learning_rate = 1e-6
# 编码器是否使用双向rnn
bidirection = True
# BeamSearch时的宽度
beam_width = 200

# keep_prob = 0.8
keep_prob = 0.5
