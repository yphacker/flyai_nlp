# -*- coding: utf-8 -*

import os
from flyai.model.base import Base
from keras.engine.saving import load_model

KERAS_MODEL_NAME = "model.h5"

from path import MODEL_PATH


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset

    def predict(self, **data):
        model = load_model(os.path.join(MODEL_PATH, KERAS_MODEL_NAME))
        data = model.predict(self.dataset.predict_data(**data))
        data = self.dataset.to_categorys(data)
        return data

    def predict_all(self, datas):
        model = load_model(os.path.join(MODEL_PATH, KERAS_MODEL_NAME))
        labels = []
        for data in datas:
            data = model.predict(self.dataset.predict_data(**data))
            data = self.dataset.to_categorys(data)
            labels.append(data)

        return labels

    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        model.save(os.path.join(path, name))

# class Model(Base):
#     def __init__(self, data):
#         self.data = data
#
#     def predict(self, path, name=KERAS_MODEL_NAME, keep='', **data):
#         model = load_model(os.path.join(path, name))
#         data = model.predict(self.data.predict_data(**data))
#         return data
#
#     def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
#         super().save_model(model, path, name, overwrite)
#         print(os.path.join(path, name))
#         model.save(os.path.join(path, name))
