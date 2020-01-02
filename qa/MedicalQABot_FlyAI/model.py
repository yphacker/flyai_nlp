# -*- coding: utf-8 -*

import os
import numpy as np
import tensorflow as tf
from flyai.model.base import Base
from tensorflow.python.saved_model import tag_constants
import config
# import seq2seq_model_config as model_config
# from seq2seq import Seq2Seq

TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.model_path = os.path.join(config.MODEL_PATH, TENSORFLOW_MODEL_DIR)

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        # model = Seq2Seq(batch_size=1, encoder_vocab_size=config.encoder_vocab_size,
        #                 decoder_vocab_size=config.decoder_vocab_size, mode='decode')
        # with tf.Session() as session:
        #     session.run(tf.global_variables_initializer())
        #     # tf.saved_model.loader.load(session, [tag_constants.SERVING], self.model_path)
        #     model.load(session, self.model_path)
        #     x_data = self.data.predict_data(**data)
        #     que_x, que_len = x_data
        #     # predict = model.predict(session, np.array(que_x), np.array(que_len))
        #     predict = model.predict(session, np.array(que_x), que_len)
        predict = ''
        return self.data.to_categorys(predict)

    def predict_all(self, datas):
        print('text len:{}'.format(len(datas)))
        for data in datas:
            print('|'.join([data[column] for column in ['que_text']]))
        preds = '' * len(datas)
        # # batch_size = 64
        # batch_size = 32
        # preds = []
        # que_x_list = []
        # que_len_list = []
        # model = Seq2Seq(batch_size=batch_size, encoder_vocab_size=config.encoder_vocab_size,
        #                 decoder_vocab_size=config.decoder_vocab_size, mode='decode')
        # with tf.Session() as session:
        #     session.run(tf.global_variables_initializer())
        #     model.load(session, self.model_path)
        #     # tf.saved_model.loader.load(session, [tag_constants.SERVING], self.model_path)
        #     for data in datas:
        #         x_data = self.data.predict_data(**data)
        #         que_x, que_len = x_data
        #         que_x_list.append(que_x)
        #         que_len_list.append(que_len)
        #         if len(que_x_list) >= batch_size:
        #             que_x_list = np.squeeze(np.array(que_x_list), axis=1)
        #             que_len_list = np.squeeze(np.array(que_len_list), axis=1)
        #             predicts = model.predict_batch(session, que_x_list, que_len_list)
        #             preds.extend([self.data.to_categorys(predict) for predict in predicts])
        #             que_x_list = []
        #             que_len_list = []
        #
        # tf.reset_default_graph()
        # batch_size = len(que_x_list)
        # model = Seq2Seq(batch_size=batch_size, encoder_vocab_size=config.encoder_vocab_size,
        #                 decoder_vocab_size=config.decoder_vocab_size, mode='decode')
        # with tf.Session() as session:
        #     session.run(tf.global_variables_initializer())
        #     model.load(session, self.model_path)
        #     # tf.saved_model.loader.load(session, [tag_constants.SERVING], self.model_path)
        #     que_x_list = np.squeeze(np.array(que_x_list), axis=1)
        #     que_len_list = np.squeeze(np.array(que_len_list), axis=1)
        #     predicts = model.predict_batch(session, que_x_list, que_len_list)
        #     preds.extend([self.data.to_categorys(predict) for predict in predicts])
        # # ans = [self.data.to_categorys(predict) for predict in preds]
        return preds

    def save_model(self, session, path, name=TENSORFLOW_MODEL_DIR, overwrite=False):
        '''
        保存模型
        :param session: 训练模型的sessopm
        :param path: 要保存模型的路径
        :param name: 要保存模型的名字
        :param overwrite: 是否覆盖当前模型
        :return:
        '''
        if overwrite:
            self.delete_file(path)

        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(path, name))
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING])
        builder.save()

    def get_tensor_name(self, name):
        return name + ":0"

    def delete_file(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
