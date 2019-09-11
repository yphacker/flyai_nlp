# coding=utf-8
# author=yphacker

import tensorflow as tf
import config
from data_helper import load_dict

word_dict, word_dict_res = load_dict()
vocab_size = max(word_dict.values()) + 1


class CNN(object):
    def __init__(self):
        # 传值空间
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # define embedding layer
        with tf.variable_scope('embedding'):
            # 标准正态分布初始化
            input_embedding = tf.Variable(
                tf.truncated_normal(shape=[vocab_size, config.embedding_dim], stddev=0.1), name='encoder_embedding')

        with tf.name_scope("cnn"):
            # CNN layer
            x_input_embedded = tf.nn.embedding_lookup(input_embedding, self.input_x)
            conv = tf.layers.conv1d(x_input_embedded, config.num_filters, config.kernel_size, name='conv')
            # global max pooling layer
            pooling = tf.reduce_max(conv, reduction_indices=[1])

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(pooling, config.dnn_dim, name='fc1')
            # fc = tf.contrib.layers.dropout(fc, keep_prob)
            fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            logits = tf.layers.dense(fc, config.numclass, name='fc2')
            y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')  # 预测类别

        with tf.name_scope("optimize"):
            # 将label进行onehot转化
            one_hot_labels = tf.one_hot(self.input_y, depth=config.numclass, dtype=tf.float32)
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

        # with tf.name_scope("summary"):
        #     tf.summary.scalar("loss", loss)
        #     tf.summary.scalar("accuracy", accuracy)
        #     merged_summary = tf.summary.merge_all()
