# -*- coding: utf-8 -*-
import json
import numpy
import os
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from path import DATA_PATH

MAX_NUM_WORDS = 30000
MAX_LEN = 512


class Transformation:
    '''
    处理训练数据的类，某些情况下需要对训练的数据再一次的处理。
    如无需处理的话，不用实现该方法。
    '''

    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        f = open(os.path.join(DATA_PATH, 'vocab.json'))
        self.tokenizer.word_index = json.loads(f.read())
        f.close()

    def transformation_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        if x_train is not None:
            texts = []
            text = self.tokenizer.texts_to_sequences(x_train)
            text = pad_sequences(text, maxlen=MAX_LEN)
            for i in range(text.shape[0]):
                split1 = numpy.split(text[i], 8)
                a = []
                for j in range(8):
                    s = numpy.split(split1[j], 8)
                    a.append(s)
                texts.append(a)
            x_train = numpy.array(texts)

        if x_test is not None:
            texts = []
            text = self.tokenizer.texts_to_sequences(x_test)
            text = pad_sequences(text, maxlen=MAX_LEN)
            for i in range(text.shape[0]):
                split1 = numpy.split(text[i], 8)
                a = []
                for j in range(8):
                    s = numpy.split(split1[j], 8)
                    a.append(s)
                texts.append(a)
            x_test = numpy.array(texts)
        return x_train, y_train, x_test, y_test
