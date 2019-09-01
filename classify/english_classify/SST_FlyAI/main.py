# -*- coding: utf-8 -*

import argparse
import numpy as np
import tensorflow as tf

try:
    from tensorflow.python.saved_model import tag_constants
except:
    from tensorflow.saved_model import tag_constants
from flyai.dataset import Dataset
from bert_model import BertModel
from model import Model
from path import MODEL_PATH

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# 模型操作辅助类
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
            model.learning_rate: learning_rate,
        }
        _loss = sess.run(model.loss, feed_dict=feed_dict)
        batch_len = len(y_val)
        total_loss += _loss * batch_len
    return total_loss / data_len


learning_rate = 5e-5
best_loss_val = 1
last_improved_step = 0
print_per_batch = 10
improvement_step = print_per_batch * 10
learning_rate_num = 0
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
            model.labels: y_train}
        if step % print_per_batch == 0:
            fetches = model.loss
            loss_train = sess.run(fetches, feed_dict=feed_dict)
            loss_val = evaluate(sess)
            if loss_val <= best_loss_val:
                best_loss_val = loss_val
                last_improved_step = step
                modelpp.save_model(sess, MODEL_PATH, overwrite=True)
                improved_str = '*'
            else:
                improved_str = ''
            cur_step = str(step + 1) + "/" + str(dataset.get_step())
            msg = 'The Current step per total: {0:>6}, Train Loss: {1:>6.2},  Val Loss: {2:>6.2},  {3}'
            print(msg.format(cur_step, loss_train, loss_val, improved_str))
        sess.run(model.train_op, feed_dict=feed_dict)
        if step - last_improved_step >= improvement_step:
            last_improved_step = step
            print("No optimization for a long time, auto adjust learning_rate...")
            learning_rate = learning_rate_decay(learning_rate)
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], MODEL_PATH)
            learning_rate_num += 1
            if learning_rate_num > 10:
                print("No optimization for a long time, auto-stopping...")
                flag = False
        if not flag:
            break
