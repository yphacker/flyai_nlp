# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
#财经
p = model.predict(text='本报讯　（记者张艳）中国华电集团昨天宣布，四川珙县电厂一期两台６０万千瓦机组正式开工建设，总投资约５０亿元。京华网ｗｗｗ．ｊｉｎｇｈｕａ．ｃｎ）（更多精彩新闻　请访问网页不支持Ｆｌａｓｈ滚动新闻栏目')
print(p)

