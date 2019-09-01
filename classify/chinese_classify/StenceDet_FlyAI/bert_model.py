# coding=utf-8
# author=yphacker

import os
import tensorflow as tf
from bert import modeling




class BertModel(object):
    def __init__(self, modelpp):
        path = modelpp.get_remote_date("https://www.flyai.com/m/multi_cased_L-12_H-768_A-12.zip")
        data_root = os.path.splitext(path)[0]
        bert_config_file = os.path.join(data_root, 'bert_config.json')
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
        bert_vocab_file = os.path.join(data_root, 'vocab.txt')

        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

        self.labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
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

        # 构建W 和 b
        output_weights = tf.get_variable(
            "output_weights", [hidden_size, num_labels],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("predict"):
            output_layer = tf.nn.dropout(output_layer, keep_prob=self.keep_prob)
            logits = tf.nn.bias_add(tf.matmul(output_layer, output_weights), output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            self.pred = tf.argmax(log_probs, 1, name='pred')

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

        with tf.name_scope("optimize"):
            # 将label进行onehot转化
            one_hot_labels = tf.one_hot(self.labels, depth=num_labels, dtype=tf.float32)
            # 构建损失函数
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss)

            # 优化器
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
