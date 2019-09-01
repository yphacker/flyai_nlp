# -*- coding: utf-8 -*
'''
实现模型的预测
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(question1='what was hour of code like (2013)?', question2='why is hour of code so popular?')
print(p)
