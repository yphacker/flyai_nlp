# -*- coding: utf-8 -*-

import numpy as np
import argparse
import tensorflow as tf
from flyai.dataset import Dataset
from model import Model
import config
# from data_helper import *

from baseline_model import LSTM
from data_helper import load_dict, process_ans_batch

# from flyai.utils.log_helper import train_log


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
args = parser.parse_args()

dataset = Dataset(batch=args.BATCH, epochs=args.EPOCHS)
modelpp = Model(dataset)

# 超参数
que_dict, ans_dict = load_dict()
encoder_vocab_size = len(que_dict)
decoder_vocab_size = len(ans_dict)
batch_size = args.BATCH

model = LSTM(batch_size)


# def evaluate(sess):
#     # val_que_x, val_que_len = que_val
#     # val_ans_x, val_ans_len = ans_val
#     # val_ans_x = process_ans_batch(val_ans_x, ans_dict, int(sorted(list(val_ans_len), reverse=True)[0]))
#     # feed_dict = {model.inputs: val_que_x,
#     #              model.targets: val_ans_x,
#     #              model.learning_rate: config.learning_rate,
#     #              model.target_sequence_length: val_ans_len,
#     #              model.source_sequence_length: val_que_len}
#     #
#     # val_loss, val_acc = sess.run([model.cost, model.ans_accuracy], feed_dict=feed_dict)
#
#     que_val_all, ans_val_all = dataset.get_all_validation_data()
#     data_len = len(ans_val_all)
#     index = np.random.permutation(len(ans_val_all))
#     n_batches = len(ans_val_all) // batch_size + 1
#     total_loss = 0.0
#     total_acc = 0.0
#     val_que_x, val_que_len = que_val_all
#     val_ans_x, val_ans_len = ans_val_all
#     for batch_index in np.array_split(index, n_batches):
#         feed_dict = {
#             model.inputs: val_que_x[batch_index],
#             model.targets: val_ans_x[batch_index],
#             model.target_sequence_length: val_ans_len[batch_index],
#             model.source_sequence_length: val_que_len[batch_index],
#             model.learning_rate: config.learning_rate,
#         }
#
#         batch_len = len(val_ans_x[batch_index])
#         _loss, _acc = sess.run([model.cost, model.ans_accuracy], feed_dict=feed_dict)
#         total_loss += _loss * batch_len
#         total_acc += _acc * batch_len
#     return total_loss / data_len, total_acc / data_len


def train():
    best_acc_val = 0
    learning_rate_num = 0
    last_improved_step = 0
    flag = True
    # with tf.Session(graph=model.train_graph) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train_writer = tf.summary.FileWriter(config.LOG_PATH, sess.graph)

        for step in range(dataset.get_step()):
            que_train, ans_train = dataset.next_train_batch()
            que_val, ans_val = dataset.next_validation_batch()
            que_x, que_length = que_train
            ans_x, ans_lenth = ans_train
            ans_x = process_ans_batch(ans_x, ans_dict, int(sorted(list(ans_lenth), reverse=True)[0]))
            feed_dict = {
                model.inputs: que_x,
                model.targets: ans_x,
                model.source_sequence_length: que_length,
                model.target_sequence_length: ans_lenth,
                model.learning_rate: config.learning_rate,
            }
            fetches = [model.train_op, model.cost, model.training_logits, model.ans_accuracy]
            _, tra_loss, logits, train_acc = sess.run(fetches, feed_dict=feed_dict)

            # summary = sess.run(summary_op, feed_dict=feed_dict)
            # train_writer.add_summary(summary, step)

            # # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
            # train_log(train_loss=tra_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

            # 实现自己的保存模型逻辑
            #     if step % 200 == 0:
            #         modelpp.save_model(sess, config.MODEL_PATH, overwrite=True)
            # modelpp.save_model(sess, config.MODEL_PATH, overwrite=True)
            if step % config.print_per_batch == 0:
                fetches = [model.cost, model.ans_accuracy]
                loss_train, acc_train = sess.run(fetches, feed_dict=feed_dict)

                val_que_x, val_que_len = que_val
                val_ans_x, val_ans_len = ans_val
                val_ans_x = process_ans_batch(val_ans_x, ans_dict, int(sorted(list(val_ans_len), reverse=True)[0]))
                feed_dict = {
                    model.inputs: val_que_x,
                    model.targets: val_ans_x,
                    model.source_sequence_length: val_que_len,
                    model.target_sequence_length: val_ans_len,
                    model.learning_rate: config.learning_rate,
                }
                loss_val, acc_val = sess.run([model.cost, model.ans_accuracy], feed_dict=feed_dict)

                # loss_val, loss_val = evaluate(sess)
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
                # learning_rate = learning_rate_decay(learning_rate)
                learning_rate_num += 1
                if learning_rate_num > 3:
                    print("No optimization for a long time, auto-stopping...")
                    flag = False
            if not flag:
                break


train()
