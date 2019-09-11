# -*- coding: utf-8 -*
import sys

import argparse
import codecs
import keras
from flyai.dataset import Dataset
from keras.layers import Input, LSTM, Dense, Embedding

import create_dict
import processor
from model import Model
from path import MODEL_PATH

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# 数据获取辅助类
dataset = Dataset()

# 模型操作辅助类
model = Model(dataset)

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
MAX_LEN = processor.MAX_LEN

rnn_unit_1 = 100  # 第一层lstm包含cell个数
rnn_unit_2 = 100  # 第二层lstm包含cell个数
conv_dim = 128
embed_dim = 100
class_num = 18
word_dict, word_dict_res = create_dict.load_dict()
num_word = max(word_dict.values()) + 1
# ——————————————————导入数据——————————————————————

input_x = Input(shape=(MAX_LEN,), dtype='int32')
x = Embedding(input_dim=num_word, output_dim=embed_dim, name='embed')(input_x)
# LSTM model
h1 = LSTM(rnn_unit_1, return_sequences=True)(x)
h2 = LSTM(rnn_unit_2)(h1)

pred = Dense(class_num, activation='softmax')(h2)
k_model = keras.Model(input_x, pred)
k_model.compile('rmsprop', 'categorical_crossentropy', ['acc', ])
for epochs in range(args.EPOCHS):
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH, test_data=False)  # 128*100
    history = k_model.fit(x_train, y_train, batch_size=args.BATCH,
                          verbose=1)
    print(str(epochs) + "/" + str(args.EPOCHS))
    if epochs % 200 == 0:
        model.save_model(k_model, MODEL_PATH, overwrite=True)
model.save_model(k_model, MODEL_PATH, overwrite=True)
