# -*- coding: utf-8 -*

import os
import tensorflow as tf
from flyai.model.base import Base
from tensorflow.python.saved_model import tag_constants
import config

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
        with tf.Session() as session:
            tf.saved_model.loader.load(session, [tag_constants.SERVING], self.model_path)
            source_input = session.graph.get_tensor_by_name(self.get_tensor_name('inputs'))
            source_seq_len = session.graph.get_tensor_by_name(self.get_tensor_name('source_sequence_length'))
            target_seq_length = session.graph.get_tensor_by_name(self.get_tensor_name('target_sequence_length'))
            predictions = session.graph.get_tensor_by_name(self.get_tensor_name('predictions'))

            x_data = self.data.predict_data(**data)
            que_x, que_len = x_data

            feed_dict = {source_input: que_x,
                         source_seq_len: que_len,
                         target_seq_length: [177] * len(que_len)}
            predict = session.run(predictions, feed_dict=feed_dict)

        return self.data.to_categorys(predict)

    def predict_all(self, datas):
        with tf.Session() as session:
            tf.saved_model.loader.load(session, [tag_constants.SERVING], self.model_path)
            source_input = session.graph.get_tensor_by_name(self.get_tensor_name('inputs'))
            source_seq_len = session.graph.get_tensor_by_name(self.get_tensor_name('source_sequence_length'))
            target_seq_length = session.graph.get_tensor_by_name(self.get_tensor_name('target_sequence_length'))
            predictions = session.graph.get_tensor_by_name(self.get_tensor_name('predictions'))

            ratings = []
            for data in datas:
                x_data = self.data.predict_data(**data)
                que_x, que_len = x_data

                feed_dict = {source_input: que_x,
                             source_seq_len: que_len,
                             target_seq_length: [177] * len(que_len)}
                predict = session.run(predictions, feed_dict=feed_dict)
                predict = self.data.to_categorys(predict)
                ratings.append(predict)
        return ratings

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
