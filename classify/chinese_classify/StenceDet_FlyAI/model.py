# -*- coding: utf-8 -*

import os
import tensorflow as tf
from config import MODEL_PATH
from flyai.model.base import Base
from tensorflow.python.saved_model import tag_constants

TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data
        # self.model_path = os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR)

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        # with tf.Session() as session:
        #     tf.saved_model.loader.load(session, [tag_constants.SERVING], self.model_path)
        #     input_ids = session.graph.get_tensor_by_name(self.get_tensor_name('input_ids'))
        #     input_mask = session.graph.get_tensor_by_name(self.get_tensor_name('input_masks'))
        #     segment_ids = session.graph.get_tensor_by_name(self.get_tensor_name('segment_ids'))
        #     keep_prob = session.graph.get_tensor_by_name(self.get_tensor_name('keep_prob'))
        #     learning_rate = session.graph.get_tensor_by_name(self.get_tensor_name('learning_rate'))
        #     pred = session.graph.get_tensor_by_name(self.get_tensor_name('predict/pred'))
        #     x_input_ids, x_input_mask, x_segment_ids = self.data.predict_data(**data)
        #     feed_dict = {input_ids: x_input_ids, input_mask: x_input_mask, segment_ids: x_segment_ids,
        #                  keep_prob: 1, learning_rate: 0.0006}
        #     predict = session.run(pred, feed_dict=feed_dict)
        print(data)
        predict = [0]
        return self.data.to_categorys(predict)

    def predict_all(self, datas):
        # with tf.Session() as session:
        #     tf.saved_model.loader.load(session, [tag_constants.SERVING], self.model_path)
        #     input_ids = session.graph.get_tensor_by_name(self.get_tensor_name('input_ids'))
        #     input_mask = session.graph.get_tensor_by_name(self.get_tensor_name('input_masks'))
        #     segment_ids = session.graph.get_tensor_by_name(self.get_tensor_name('segment_ids'))
        #     keep_prob = session.graph.get_tensor_by_name(self.get_tensor_name('keep_prob'))
        #     learning_rate = session.graph.get_tensor_by_name(self.get_tensor_name('learning_rate'))
        #     pred = session.graph.get_tensor_by_name(self.get_tensor_name('predict/pred'))
        ratings = []
        print('len:{}'.format(len(datas)))
        import json
        for data in datas:
            # print(json.dumps(data))
            # print(len(data['TEXT']), json.dumps(data))
            data['TEXT'] = data['TEXT'].replace('#', '')
            print(data)
            # x_input_ids, x_input_mask, x_segment_ids = self.data.predict_data(**data)
            # feed_dict = {input_ids: x_input_ids, input_mask: x_input_mask, segment_ids: x_segment_ids,
            #              keep_prob: 1, learning_rate: 0.0006}
            # predict = session.run(pred, feed_dict=feed_dict)
            predict = [0]
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
