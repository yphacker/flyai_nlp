# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
# import tensorflow as tf
from flyai.dataset import Dataset
from model import Model
# import config
# # from baseline_model import LSTM
# import seq2seq_model_config as model_config
# from seq2seq import Seq2Seq
# from data_helper import load_dict, process_ans_batch

# from flyai.utils.log_helper import train_log


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
parser.add_argument("-e", "--EPOCHS", default=64, type=int, help="train epochs")
args = parser.parse_args()

dataset = Dataset(batch=args.BATCH, val_batch=args.BATCH, epochs=args.EPOCHS)
modelpp = Model(dataset)

# # 超参数
# que_dict, ans_dict = load_dict()
# encoder_vocab_size = len(que_dict)
# decoder_vocab_size = len(ans_dict)
# batch_size = args.BATCH

x_train, y_train, x_val, y_val = dataset.get_all_data()
all_x = np.concatenate([x_train, x_val])
all_y = np.concatenate([y_train, y_val])
x = pd.DataFrame([var for var in all_x])
y = pd.DataFrame([var for var in all_y])
train = pd.concat([x, y], axis=1)
print('train len:{}'.format(train.shape[0]))
# for i, row in train.iterrows():
#     print('|'.join([row[column] for column in ['que_text', 'ans_text']]))


# print(train.head())

# model = LSTM(batch_size)
# model = Seq2Seq(batch_size=batch_size, encoder_vocab_size=encoder_vocab_size, decoder_vocab_size=decoder_vocab_size,
#                 mode='train')


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
#     data_len = 0
#     # index = np.random.permutation(len(ans_val_all))
#     # n_batches = len(ans_val_all) // batch_size + 1
#     total_loss = 0.0
#     total_acc = 0.0
#     val_que_x, val_que_len = que_val_all
#     val_ans_x, val_ans_len = ans_val_all
#     n_batches = len(val_que_x) // batch_size
#     for i in range(n_batches):
#         que_x = val_que_x[i * batch_size: (i + 1) * batch_size]
#         que_length = val_que_len[i * batch_size: (i + 1) * batch_size]
#         ans_x = val_ans_x[i * batch_size: (i + 1) * batch_size]
#         ans_lenth = val_ans_len[i * batch_size: (i + 1) * batch_size]
#         ans_x = process_ans_batch(ans_x, ans_dict, int(sorted(list(ans_lenth), reverse=True)[0]))
#         _loss, _lr = model.train(sess, que_x, que_length, ans_x, ans_lenth, model_config.keep_prob)
#         total_loss += _loss * batch_size
#     return total_loss / data_len, total_acc / data_len
#
#
# def train():
#     # best_acc_val = 0
#     best_loss_val = 100
#     learning_rate_num = 0
#     last_improved_step = 0
#     flag = True
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         # train_writer = tf.summary.FileWriter(config.LOG_PATH, sess.graph)
#
#         for step in range(dataset.get_step()):
#             que_train, ans_train = dataset.next_train_batch()
#             que_val, ans_val = dataset.next_validation_batch()
#             dataset.next_batch()
#             que_x, que_length = que_train
#             ans_x, ans_lenth = ans_train
#             ans_x = process_ans_batch(ans_x, ans_dict, int(sorted(list(ans_lenth), reverse=True)[0]))
#             loss_train, lr = model.train(sess, np.asarray(que_x), que_length, np.asarray(ans_x), ans_lenth,
#                                          model_config.keep_prob)
#             # summary = sess.run(summary_op, feed_dict=feed_dict)
#             # train_writer.add_summary(summary, step)
#
#             # # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
#             # train_log(train_loss=tra_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
#
#             # 实现自己的保存模型逻辑
#             #     if step % 200 == 0:
#             #         modelpp.save_model(sess, config.MODEL_PATH, overwrite=True)
#             # modelpp.save_model(sess, config.MODEL_PATH, overwrite=True)
#             if (step + 1) % config.print_per_batch == 0:
#
#                 val_que_x, val_que_len = que_val
#                 val_ans_x, val_ans_len = ans_val
#                 val_ans_x = process_ans_batch(val_ans_x, ans_dict, int(sorted(list(val_ans_len), reverse=True)[0]))
#                 loss_val, lr = model.train(sess, np.asarray(val_que_x), val_que_len,
#                                            np.asarray(val_ans_x), val_ans_len, 1)
#
#                 # loss_val, acc_val = evaluate(sess)
#                 if loss_val < best_loss_val:
#                     best_loss_val = loss_val
#                     last_improved_step = step
#                     # modelpp.save_model(sess, config.MODEL_PATH, overwrite=True)
#                     import os
#                     if not os.path.exists(config.MODEL_PATH):
#                         os.makedirs(config.MODEL_PATH)
#                     model_path = os.path.join(config.MODEL_PATH, 'best')
#                     model.save(sess, model_path)
#                     improved_str = '*'
#                 else:
#                     improved_str = ''
#                 cur_step = str(step + 1) + "/" + str(dataset.get_step())
#                 msg = 'the current step: {0:>6}, Train Loss: {1:>6.2}, Val Loss: {2:>6.2}, {3}'
#                 print(msg.format(cur_step, loss_train, loss_val, improved_str))
#             if step - last_improved_step >= config.improvement_step:
#                 last_improved_step = step
#                 print("No optimization for a long time, auto adjust learning_rate...")
#                 learning_rate_num += 1
#                 if learning_rate_num > 3:
#                     print("No optimization for a long time, auto-stopping...")
#                     flag = False
#             if not flag:
#                 break
#
#
# train()
