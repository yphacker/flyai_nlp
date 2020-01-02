# coding=utf-8
# author=yphacker

import os
import tensorflow as tf
from bert import modeling
import config
from flyai.utils import remote_helper
from model_utils import focal_loss


class BertModel(object):
    def __init__(self):
        # path = remote_helper.get_remote_date('https://www.flyai.com/m/roberta_chinese_wwm_ext_L-12_H-768_A-12.zip')
        remote_helper.get_remote_date(config.pretrain_model_url_path)
        bert_config_file = os.path.join(config.pretrain_model_path, 'bert_config.json')
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        init_checkpoint = os.path.join(config.pretrain_model_path, 'bert_model.ckpt')
        bert_vocab_file = os.path.join(config.pretrain_model_path, 'vocab.txt')

        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        self.labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
        # self.labels = tf.placeholder(tf.float32, shape=[None, config.num_labels], name='labels')

        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

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
            "output_weights", [hidden_size, config.num_labels],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [config.num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("predict"):
            if self.is_training is True:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.5)
            logits = tf.nn.bias_add(tf.matmul(output_layer, output_weights), output_bias)
            probs = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            self.pred = tf.argmax(log_probs, 1, name='pred')

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
            # correct_pred = tf.equal(tf.argmax(self.labels, 1), self.pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

        with tf.name_scope("optimize"):
            # 必须指明batch
            # per_example_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            # self.loss = tf.reduce_mean(per_example_loss)
            # 将label进行onehot转化
            one_hot_labels = tf.one_hot(self.labels, depth=config.num_labels, dtype=tf.float32)
            # # 构建损失函数
            # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            # 1,10,20,40,200
            per_example_loss = -tf.reduce_sum(0.01 * one_hot_labels[:, 0] * log_probs[:, 0]
                                              + 0.1 * one_hot_labels[:, 1] * log_probs[:, 1]
                                              + 0.2 * one_hot_labels[:, 2] * log_probs[:, 2]
                                              + 0.4 * one_hot_labels[:, 3] * log_probs[:, 3]
                                              # + 2 * one_hot_labels[:, 4] * log_probs[:, 4])
                                              + 2.5 * one_hot_labels[:, 4] * log_probs[:, 4])
            self.loss = tf.reduce_mean(per_example_loss)
            #
            # self.loss = focal_loss(probs, one_hot_labels)
            # 优化器
            # self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

            # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
            # self.learning_rate_=learning_rate
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
            # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            #     self.train_op = optimizer.apply_gradients(zip(gradients, variables))
