# coding=utf-8
# author=yphacker

import argparse
import numpy as np
import tensorflow as tf
from flyai.dataset import Dataset
import config
from model import Model
from cnn_model import CNN

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

dataset = Dataset(batch=args.BATCH, epochs=args.EPOCHS)
modelpp = Model(dataset)
model = CNN()


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
#
#     # dataset.get_step() 获取数据的总迭代次数
#     for step in range(dataset.get_step()):
#         x_train, y_train = dataset.next_train_batch()
#         x_val, y_val = dataset.next_validation_batch()
#
#         fetches = [loss, accuracy, train_op]
#         feed_dict = {input_x: x_train, input_y: y_train, keep_prob: 0.5}
#         loss_, accuracy_, _ = sess.run(fetches, feed_dict=feed_dict)
#
#         valid_acc = sess.run(accuracy, feed_dict={input_x: x_val, input_y: y_val, keep_prob: 1.0})
#         summary = sess.run(merged_summary, feed_dict=feed_dict)
#         train_writer.add_summary(summary, step)
#
#         cur_step = str(step + 1) + "/" + str(dataset.get_step())
#         print('The Current step per total: {} | The Current loss: {} | The Current ACC: {} |'
#               ' The Current Valid ACC: {}'.format(cur_step, loss_, accuracy_, valid_acc))
#         if step % 100 == 0:  # 每隔100个step存储一次model文件
#             model.save_model(sess, MODEL_PATH, overwrite=True)

def learning_rate_decay(learning_rate):
    return learning_rate * 0.5


def evaluate(sess):
    x_val, y_val = dataset.get_all_validation_data()
    data_len = len(y_val)
    index = np.random.permutation(len(y_val))
    n_batches = len(y_val) // args.BATCH + 1
    total_loss = 0.0
    total_acc = 0.0
    for batch_index in np.array_split(index, n_batches):
        feed_dict = {model.input_x: x_val[batch_index], model.input_y: y_val[batch_index], model.keep_prob: 1}
        _loss, _acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        batch_len = len(y_val[batch_index])
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
        # train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
        print('dataset.get_step:', dataset.get_step())
        for step in range(dataset.get_step()):
            x_train, y_train = dataset.next_train_batch()
            if step % config.print_per_batch == 0:
                fetches = [model.loss, model.accuracy]
                feed_dict = {model.input_x: x_train, model.input_y: y_train, model.keep_prob: 1}
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
                msg = 'The Current step per total: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%} {5}'
                print(msg.format(cur_step, loss_train, acc_train, loss_val, acc_val, improved_str))
            feed_dict = {model.input_x: x_train, model.input_y: y_train, model.keep_prob: 0.5}
            sess.run(model.train_op, feed_dict=feed_dict)
            # 验证集正确率长期不提升，提前结束训练
            if step - last_improved_step >= config.improvement_step:
                last_improved_step = step
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto adjust learning_rate...")
                learning_rate = learning_rate_decay(learning_rate)
                learning_rate_num += 1
                if learning_rate_num > 5:
                    print("No optimization for a long time, auto-stopping...")
                    flag = False
            if not flag:
                break


train()
