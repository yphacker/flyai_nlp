# -*- coding: utf-8 -*

import argparse
import numpy as np
import tensorflow as tf
from flyai.dataset import Dataset
import config
from config import MODEL_PATH
from model import Model
from lstm_ner_model import NER_net
from utils import load_word2vec_embedding

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=32, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# 模型操作辅助类
modelpp = Model(dataset)

# 训练神经网络

embedding = load_word2vec_embedding(config.vocab_size)
model = NER_net(embedding, args.BATCH)


def evaluate(sess):
    """评估在某一数据上的准确率和损失"""
    x_val_all, y_val_all = dataset.get_all_validation_data()
    data_len = len(y_val_all)
    index = np.random.permutation(len(y_val_all))
    n_batches = len(y_val_all) // args.BATCH + 1
    total_loss = 0.0
    for batch_index in np.array_split(index, n_batches):
        x_val = x_val_all[batch_index]
        y_val = y_val_all[batch_index]
        max_sentenc_length = max(map(len, x_val))
        sequence_len = np.asarray([len(x) for x in x_val])
        # padding
        x_val = np.asarray([list(x[:]) + (max_sentenc_length - len(x)) * [config.src_padding] for x in x_val])
        y_val = np.asarray([list(y[:]) + (max_sentenc_length - len(y)) * [TAGS_NUM - 1] for y in y_val])
        feed_dict = {model.input: x_val, model.label: y_val, model.seq_length: sequence_len}
        _loss = sess.run(model.loss, feed_dict=feed_dict)
        batch_len = len(x_val)
        total_loss += _loss * batch_len
    return total_loss / data_len


TAGS_NUM = config.label_len
best_loss_val = 1e5
last_improved_step = 0
print_per_batch = 10
improvement_step = print_per_batch * 10
flag = True
with tf.Session() as sess:
    # merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    # writer = tf.summary.FileWriter(LOG_PATH, sess.graph)  # 将训练日志写入到logs文件夹下
    sess.run(tf.global_variables_initializer())
    print(dataset.get_step())
    for step in range(dataset.get_step()):
        x_train, y_train = dataset.next_train_batch()
        max_sentenc_length = max(map(len, x_train))
        sequence_len = np.asarray([len(x) for x in x_train])
        # padding
        x_train = np.asarray([list(x[:]) + (max_sentenc_length - len(x)) * [config.src_padding] for x in x_train])
        y_train = np.asarray([list(y[:]) + (max_sentenc_length - len(y)) * [TAGS_NUM - 1] for y in y_train])
        # res, loss_, _ = sess.run([merged, net.loss, net.train_op],
        #                          feed_dict={net.input: x_train, net.label: y_train, net.seq_length: sequence_len})
        # print('steps:{}loss:{}'.format(i, loss_))
        # writer.add_summary(res, i)  # 将日志数据写入文件
        # if i % 50 == 0:
        #     modelpp.save_model(sess, MODEL_PATH, overwrite=True)
        feed_dict = {model.input: x_train, model.label: y_train, model.seq_length: sequence_len}
        if step % print_per_batch == 0:
            # fetches = [model.loss]
            loss_train = sess.run(model.loss, feed_dict=feed_dict)
            loss_val = evaluate(sess)
            if loss_val <= best_loss_val:
                best_loss_val = loss_val
                last_improved_step = step
                modelpp.save_model(sess, MODEL_PATH, overwrite=True)
                improved_str = '*'
            else:
                improved_str = ''
            cur_step = str(step + 1) + "/" + str(dataset.get_step())
            msg = 'The current step: {0:>6}, Train Loss: {1:>6.2}, Val Loss: {2:>6.2} {3}'
            print(msg.format(cur_step, loss_train, loss_val, improved_str))
        sess.run(model.train_op, feed_dict=feed_dict)
        if step - last_improved_step >= improvement_step:
            last_improved_step = step
            print("No optimization for a long time, auto adjust learning_rate...")
            flag = False
        if not flag:
            break
