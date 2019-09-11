# -*- coding: utf-8 -*
import argparse
import json
import numpy
import os
from flyai.dataset import Dataset
from keras import Input, Model
from keras.layers import Embedding, Dropout, GRU, TimeDistributed, Dense
from keras.optimizers import Adam

import model
from path import DATA_PATH, MODEL_PATH

# 设置超参数
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 100
NUM_FILTERS = 50
MAX_LEN = 512
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

# 加载经过预训练的 GloVe 词嵌入模型
glove_path = os.path.join(DATA_PATH, 'glove.txt')
embeddings_index = {}
f = open(glove_path, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# 使用经过预训练的GloVe词嵌入将embedding层初始化
embedding_matrix = numpy.random.random((MAX_NUM_WORDS + 1, EMBEDDING_DIM))
f = open(os.path.join(DATA_PATH, 'vocab.json'))
vocab = json.loads(f.read())
f.close()
for word, i in vocab.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # 没有出现在 embedding index中的词汇将会随机初始化（Random Initialized）
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(MAX_NUM_WORDS + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=int(MAX_LEN // 64),
                            trainable=True)

input1 = Input(shape=(int(MAX_LEN // 64),), dtype='int32')
embed = embedding_layer(input1)
gru1 = GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed)
gru1 = Dropout(0.5)(gru1)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8, int(MAX_LEN // 64),), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed2)
gru2 = Dropout(0.5)(gru2)
Encoder2 = Model(input2, gru2)

input3 = Input(shape=(8, 8, int(MAX_LEN // 64)), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed3)
gru3 = Dropout(0.5)(gru3)
preds = Dense(5, activation='softmax')(gru3)
sqeue = Model(input3, preds)

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
sqeue.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
sqeue.summary()

# 数据获取辅助类
dataset = Dataset()
save = model.Model(dataset)
for epochs in range(args.EPOCHS):
    # 得到训练和测试的数据
    x_train, y_train, _, _ = dataset.next_batch(args.BATCH, test_data=False)
    history = sqeue.fit(x_train, y_train, batch_size=args.BATCH)
    if epochs % 5 == 0:
        print(str(epochs) + "/" + str(args.EPOCHS))
        save.save_model(sqeue, MODEL_PATH, overwrite=True)
