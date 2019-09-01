# -*- coding: utf-8 -*
from flyai.dataset import Dataset
from model import Model

# 数据获取辅助类
dataset = Dataset()
# 模型操作辅助类
model = Model(dataset)

result = model.predict(text="link lets you pick a city bomb size and altitude of detonation. ace!")
print(result)