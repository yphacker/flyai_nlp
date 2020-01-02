# coding=utf-8
# author=yphacker

import os
import numpy as np
import tensorflow as tf
from flyai.model.base import Base
from config import MODEL_PATH
from tensorflow.python.saved_model import tag_constants

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
            tf.saved_model.loader.load(session, [tag_constants.SERVING], os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
            # input_ids = session.graph.get_tensor_by_name(self.get_tensor_name('input'))
            # output = session.graph.get_tensor_by_name(self.get_tensor_name('output'))
            # transition_params = session.graph.get_tensor_by_name(self.get_tensor_name('transition_params'))
            # x_input_ids = self.data.predict_data(**data)
            # tf_unary_scores, tf_transition_params = session.run([output, transition_params],
            #                                                     feed_dict={input_ids: x_input_ids})
            # # 把batch那个维度去掉
            # tf_unary_scores = np.squeeze(tf_unary_scores)
            # viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores, tf_transition_params)

            input_ids = session.graph.get_tensor_by_name(self.get_tensor_name('input_ids'))
            input_mask = session.graph.get_tensor_by_name(self.get_tensor_name('input_masks'))
            segment_ids = session.graph.get_tensor_by_name(self.get_tensor_name('segment_ids'))
            pred = session.graph.get_tensor_by_name(self.get_tensor_name('predict/pred'))
            x_input_ids, x_input_mask, x_segment_ids = self.data.predict_data(**data)
            feed_dict = {input_ids: x_input_ids, input_mask: x_input_mask,segment_ids: x_segment_ids}
            predict = session.run(pred, feed_dict=feed_dict)
            return self.data.to_categorys(predict)

    def predict_all(self, datas):
        with tf.Session() as session:
            tf.saved_model.loader.load(session, [tag_constants.SERVING], os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
            # input_ids = session.graph.get_tensor_by_name(self.get_tensor_name('input'))
            # output = session.graph.get_tensor_by_name(self.get_tensor_name('output'))
            # transition_params = session.graph.get_tensor_by_name(self.get_tensor_name('transition_params'))
            input_ids = session.graph.get_tensor_by_name(self.get_tensor_name('input_ids'))
            input_mask = session.graph.get_tensor_by_name(self.get_tensor_name('input_masks'))
            segment_ids = session.graph.get_tensor_by_name(self.get_tensor_name('segment_ids'))
            pred = session.graph.get_tensor_by_name(self.get_tensor_name('predict/pred'))
            ratings = []
            for data in datas:
                # x_input_ids = self.data.predict_data(**data)
                # tf_unary_scores, tf_transition_params = session.run([output, transition_params],
                #                                                     feed_dict={input_ids: x_input_ids})
                # # 把batch那个维度去掉
                # tf_unary_scores = np.squeeze(tf_unary_scores)
                #
                # viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores, tf_transition_params)
                x_input_ids, x_input_mask, x_segment_ids = self.data.predict_data(**data)
                feed_dict = {input_ids: x_input_ids, input_mask: x_input_mask, segment_ids: x_segment_ids}
                predict = session.run(pred, feed_dict=feed_dict)
                ratings.append(self.data.to_categorys(predict))
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
