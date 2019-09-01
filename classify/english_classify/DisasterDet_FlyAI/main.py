# -*- coding: utf-8 -*-

import argparse
import numpy as np
import tensorflow as tf
from flyai.dataset import Dataset
import config
from model import Model
from bert_model import BertModel

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
args = parser.parse_args()

dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
modelpp = Model(dataset)
model = BertModel(modelpp)


def learning_rate_decay(learning_rate):
    return learning_rate * 0.5


def evaluate(sess):
    """评估在某一数据上的准确率和损失"""
    x_val_all, y_val_all = dataset.get_all_validation_data()
    data_len = len(y_val_all)
    index = np.random.permutation(len(y_val_all))
    n_batches = len(y_val_all) // args.BATCH + 1
    total_loss = 0.0
    total_acc = 0.0
    x_input_ids_val = x_val_all[0]
    x_input_mask_val = x_val_all[1]
    x_segment_ids_val = x_val_all[2]
    for batch_index in np.array_split(index, n_batches):
        x_input_ids = x_input_ids_val[batch_index]
        x_input_mask = x_input_mask_val[batch_index]
        x_segment_ids = x_segment_ids_val[batch_index]
        y_val = y_val_all[batch_index]
        feed_dict = {
            model.input_ids: x_input_ids,
            model.input_mask: x_input_mask,
            model.segment_ids: x_segment_ids,
            model.labels: y_val,
            model.is_training: False,
        }
        batch_len = len(y_val)
        _loss, _acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += _loss * batch_len
        total_acc += _acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train():
    best_acc_val = 0
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
                model.input_ids: x_input_ids,
                model.input_mask: x_input_mask,
                model.segment_ids: x_segment_ids,
                model.labels: y_train,
                model.learning_rate: learning_rate,
                model.is_training: True,
            }
            if step % config.print_per_batch == 0:
                fetches = [model.loss, model.accuracy]
                loss_train, acc_train = sess.run(fetches, feed_dict=feed_dict)
                loss_val, acc_val = evaluate(sess)
                if acc_val >= best_acc_val:
                    best_acc_val = acc_val
                    last_improved_step = step
                    modelpp.save_model(sess, config.MODEL_PATH, overwrite=True)
                    improved_str = '*'
                else:
                    improved_str = ''
                cur_step = str(step + 1) + "/" + str(dataset.get_step())
                msg = 'the Current step: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, {5}'
                print(msg.format(cur_step, loss_train, acc_train, loss_val, acc_val, improved_str))
            sess.run(model.train_op, feed_dict=feed_dict)
            if step - last_improved_step >= config.improvement_step:
                last_improved_step = step
                print("No optimization for a long time, auto adjust learning_rate...")
                learning_rate = learning_rate_decay(learning_rate)
                learning_rate_num += 1
                if learning_rate_num > 5:
                    print("No optimization for a long time, auto-stopping...")
                    flag = False
            if not flag:
                break


train()
