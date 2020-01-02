# coding=utf-8
# author=yphacker

import os
import numpy as np
import tensorflow as tf
from flyai.utils import remote_helper
import config
from bert import modeling
from bilstm_crf import BiLstmCrf
import bert_model_config as model_config


class BertLstmNer(object):
    def __init__(self):
        path = remote_helper.get_remote_date("https://www.flyai.com/m/uncased_L-24_H-1024_A-16.zip")
        data_root = os.path.splitext(path)[0]
        bert_config_file = os.path.join(data_root, 'bert_config.json')
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
        bert_vocab_file = os.path.join(data_root, 'vocab.txt')

        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        self.labels = tf.placeholder(tf.int32, shape=[None, ], name="labels")

        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.learning_rate = tf.placeholder_with_default(config.learning_rate, shape=(), name='learning_rate')

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
            # output_layer = model.get_pooled_output()  # 这个获取句子的output
            # hidden_size = output_layer.shape[-1].value  # 获取输出的维度
            embedding = model.get_sequence_output()
            max_seq_length = embedding.shape[1].value

        used = tf.sign(tf.abs(self.input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        blstm_crf = BiLstmCrf(embedded_chars=embedding, max_seq_length=max_seq_length, labels=self.labels,
                              lengths=lengths, is_training=self.is_training)

        self.loss, logits, trans, pred_ids = blstm_crf.add_blstm_crf_layer()

        with tf.variable_scope("predict"):
            self.pred = tf.Variable(pred_ids, name='pred')

        with tf.name_scope("train_op"):
            self.train_op = tf.train.AdamOptimizer(learning_rate=model_config.learning_rate).minimize(self.loss)

    def train(self, dataset, modelpp):
        best_loss_val = 1e2
        last_improved_step = 0
        learning_rate_num = 0
        learning_rate = config.learning_rate
        flag = True
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('dataset.get_step:', dataset.get_step())
            for step in range(dataset.get_step()):
                x_train, y_train = dataset.next_train_batch()
                x_input_ids = x_train[0]
                x_input_mask = x_train[1]
                x_segment_ids = x_train[2]
                feed_dict = {
                    self.input_ids: x_input_ids,
                    self.input_mask: x_input_mask,
                    self.segment_ids: x_segment_ids,
                    self.labels: y_train,
                    self.learning_rate: learning_rate,
                    self.is_training: True,
                }
                sess.run(self.train_op, feed_dict=feed_dict)
                if (step + 1) % config.print_per_batch == 0:
                    loss_train = sess.run(self.loss, feed_dict=feed_dict)
                    loss_val = self.evaluate(sess, dataset)
                    if loss_val <= best_loss_val:
                        best_loss_val = loss_val
                        last_improved_step = step
                        modelpp.save_model(sess, config.MODEL_PATH, overwrite=True)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    cur_step = str(step + 1) + "/" + str(dataset.get_step())
                    msg = 'The current step: {0:>6}, Train Loss: {1:>6.2}, Val Loss: {2:>6.2} {3}'
                    print(msg.format(cur_step, loss_train, loss_val, improved_str))
                if step - last_improved_step >= config.improvement_step:
                    last_improved_step = step
                    print("No optimization for a long time, auto adjust learning_rate...")
                    # learning_rate = learning_rate_decay(learning_rate)
                    learning_rate_num += 1
                    if learning_rate_num > 3:
                        print("No optimization for a long time, auto-stopping...")
                        flag = False
                if not flag:
                    break

    def evaluate(self, sess, dataset):
        """评估在某一数据上的准确率和损失"""
        x_val_all, y_val_all = dataset.get_all_validation_data()
        data_len = len(y_val_all)
        index = np.random.permutation(len(y_val_all))
        n_batches = len(y_val_all) // config.batch_size + 1
        total_loss = 0.0
        x_input_ids_val = x_val_all[0]
        x_input_mask_val = x_val_all[1]
        x_segment_ids_val = x_val_all[2]
        for batch_index in np.array_split(index, n_batches):
            x_input_ids = x_input_ids_val[batch_index]
            x_input_mask = x_input_mask_val[batch_index]
            x_segment_ids = x_segment_ids_val[batch_index]
            y_val = y_val_all[batch_index]
            feed_dict = {
                self.input_ids: x_input_ids,
                self.input_mask: x_input_mask,
                self.segment_ids: x_segment_ids,
                self.labels: y_val,
                self.is_training: False,
            }
            batch_len = len(y_val)
            _loss, _acc = sess.run(self.loss, feed_dict=feed_dict)
            total_loss += _loss * batch_len
        return total_loss / data_len
