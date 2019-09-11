# -*- coding: utf-8 -*-
import numpy as np
class Transformation:
    '''
    处理训练数据的类，某些情况下需要对训练的数据再一次的处理。
    如无需处理的话，不用实现该方法。
    '''

    def transformation_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        # max_seq_len=20
        # input_size=10
        #
        # if x_train is not None:
        #     x_train = x_train
        #     y_train = np.reshape(y_train, (-1, 1))
        #
        # if x_test is not None:
        #     x_test = x_test
        #     y_test = np.reshape(y_test, (-1, 1))

        return x_train, y_train, x_test, y_test
