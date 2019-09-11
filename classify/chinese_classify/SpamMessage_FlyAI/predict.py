# -*- coding: utf-8 -*
from flyai.dataset import Dataset
from model import Model

# 数据获取辅助类
dataset = Dataset()
# 模型操作辅助类
model = Model(dataset)

result = model.predict(text="全体裁判和工作人员都用热烈的掌声祝贺我们胜利归来")
print(result)
