# coding=utf-8
# author=yphacker

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
import config


class NerNet:
    def __init__(self, embedding, batch_size=32):
        '''
        :param scope_name:
        :param iterator: 调用tensorflow DataSet API把数据feed进来。
        :param embedding: 提前训练好的word embedding
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.embedding = embedding
        # ——————————————————导入数据——————————————————————
        self.input = tf.placeholder(tf.int32, shape=[None, None], name="input")
        self.label = tf.placeholder(tf.int32, shape=[None, None], name="label")
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name="seq_length_in_batch")
        self._build_net()

    def _build_net(self):
        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.nn.embedding_lookup(self.embedding, self.input)
        # y: [batch_size, time_step]
        self.y = self.label

        cell_forward = tf.contrib.rnn.BasicLSTMCell(config.lstm_size)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(config.lstm_size)
        if config.keep_prob is not None:
            cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=config.keep_prob)
            cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=config.keep_prob)

        # time_major 可以适应输入维度。
        outputs, bi_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, dtype=tf.float32)

        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)

        # projection:
        W = tf.get_variable("projection_w", [2 * config.lstm_size, config.num_labels])
        b = tf.get_variable("projection_b", [config.num_labels])
        x_reshape = tf.reshape(outputs, [-1, 2 * config.lstm_size])
        projection = tf.add(tf.matmul(x_reshape, W), b, name='projection')
        nsteps = tf.shape(outputs)[1]
        # -1 to time step
        self.outputs = tf.reshape(projection, [-1, nsteps, config.num_labels], name='output')

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, self.seq_length)
        self.transition_params = tf.add(self.transition_params, 0, name='transition_params')
        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
        # tf.summary.scalar('loss', self.loss)

    def train(self, dataset, modelpp):
        best_loss_val = 100
        last_improved_step = 0
        flag = True
        with tf.Session() as sess:
            # merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
            # writer = tf.summary.FileWriter(LOG_PATH, sess.graph)  # 将训练日志写入到logs文件夹下
            sess.run(tf.global_variables_initializer())
            print(dataset.get_step())
            for step in range(dataset.get_step()):
                x_train, y_train = dataset.next_train_batch()
                max_sentenc_length = max(map(len, x_train))
                seq_length = np.asarray([len(x) for x in x_train])
                # padding
                x_train = np.asarray(
                    [list(x[:]) + (max_sentenc_length - len(x)) * [config.src_padding] for x in x_train])
                y_train = np.asarray(
                    [list(y[:]) + (max_sentenc_length - len(y)) * [config.num_labels - 1] for y in y_train])
                # writer.add_summary(res, i)  # 将日志数据写入文件
                feed_dict = {self.input: x_train, self.label: y_train, self.seq_length: seq_length}
                if step % config.print_per_batch == 0:

                    # fetches = [model.loss]
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
                sess.run(self.train_op, feed_dict=feed_dict)
                if step - config.last_improved_step >= config.improvement_step:
                    last_improved_step = step
                    print("No optimization for a long time, auto adjust learning_rate...")
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
        for batch_index in np.array_split(index, n_batches):
            x_val = x_val_all[batch_index]
            y_val = y_val_all[batch_index]
            max_sentenc_length = max(map(len, x_val))
            sequence_len = np.asarray([len(x) for x in x_val])
            # padding
            x_val = np.asarray([list(x[:]) + (max_sentenc_length - len(x)) * [config.src_padding] for x in x_val])
            y_val = np.asarray([list(y[:]) + (max_sentenc_length - len(y)) * [config.num_labels - 1] for y in y_val])
            feed_dict = {self.input: x_val, self.label: y_val, self.seq_length: sequence_len}
            _loss = sess.run(self.loss, feed_dict=feed_dict)
            batch_len = len(x_val)
            total_loss += _loss * batch_len
        return total_loss / data_len
