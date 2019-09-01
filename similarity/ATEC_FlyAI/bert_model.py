# coding=utf-8
# author=yphacker

import os
import tensorflow as tf
from bert import modeling


class BertModel(object):
    def __init__(self, modelpp):
        path = modelpp.get_remote_date("https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip")
        data_root = os.path.splitext(path)[0]
        bert_config_file = os.path.join(data_root, 'bert_config.json')
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
        bert_vocab_file = os.path.join(data_root, 'vocab.txt')

        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

        self.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # 创建bert模型
        with tf.name_scope('Bert'):
            model = modeling.BertModel(
                config=bert_config,
                is_training=True,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
                use_one_hot_embeddings=False
            )
            # 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
            # output_layer = model.get_sequence_output()
            tvars = tf.trainable_variables()
            # 加载BERT模型
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            output_layer = model.get_pooled_output()  # 这个获取句子的output
            hidden_size = output_layer.shape[-1].value  # 获取输出的维度

        # ——————————————————定义神经网络变量——————————————————
        # 输入层、输出层权重、偏置
        weights = {
            'out': tf.Variable(tf.random_normal([768, 1]))
        }
        biases = {
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }
        w_out = weights['out']
        b_out = biases['out']
        logits = tf.add(tf.matmul(output_layer, w_out), b_out)
        self.pred = tf.reshape(logits, shape=[-1, 1], name="pred")

        # # 损失函数
        self.loss = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.labels, [-1])))
        # tf.summary.scalar('loss', loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
