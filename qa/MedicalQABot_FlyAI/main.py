# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
from flyai.dataset import Dataset
from model import Model
import config
# from data_helper import *
from tensorflow.python.layers.core import Dense
from baseline_model import LSTM
from data_helper import load_dict, process_ans_batch
# from flyai.utils.log_helper import train_log


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()

dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = LSTM()
modelpp = Model(dataset)




# 超参数
que_dict, ans_dict = load_dict()
encoder_vocab_size = len(que_dict)
decoder_vocab_size = len(ans_dict)
# Batch Size,
batch_size = args.BATCH


with tf.Session(graph=model.train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(config.LOG_PATH, sess.graph)

    for step in range(dataset.get_step()):
        que_train, ans_train = dataset.next_train_batch()
        que_val, ans_val = dataset.next_validation_batch()

        que_x, que_length = que_train
        ans_x, ans_lenth = ans_train
        ans_x = process_ans_batch(ans_x, ans_dict, int(sorted(list(ans_lenth), reverse=True)[0]))

        feed_dict = {
            model.input_data: que_x,
             model.targets: ans_x,
             model.lr: config.learning_rate,
             model.target_sequence_length: ans_lenth,
             model.source_sequence_length: que_length
        }
        fetches = [model.train_op, cost, training_logits, ans_accuracy]
        _, tra_loss, logits, train_acc = sess.run(fetches, feed_dict=feed_dict)

        val_que_x, val_que_len = que_val
        val_ans_x, val_ans_len = ans_val
        val_ans_x = process_ans_batch(val_ans_x, ans_dict, int(sorted(list(val_ans_len), reverse=True)[0]))
        feed_dict = {input_data: val_que_x,
                     targets: val_ans_x,
                     lr: learning_rate,
                     target_sequence_length: val_ans_len,
                     source_sequence_length: val_que_len}

        val_loss, val_acc = sess.run([cost, ans_accuracy], feed_dict=feed_dict)

        summary = sess.run(summary_op, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        # # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
        # train_log(train_loss=tra_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

        # 实现自己的保存模型逻辑
        if step % 200 == 0:
            model.save_model(sess, MODEL_PATH, overwrite=True)
    model.save_model(sess, MODEL_PATH, overwrite=True)
