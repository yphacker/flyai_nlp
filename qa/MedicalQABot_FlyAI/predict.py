# -*- coding: utf-8 -*
'''
实现模型的调用
'''
import jieba
import pandas as pd
from flyai.dataset import Dataset
from model import Model
from nltk.translate.bleu_score import sentence_bleu

data = Dataset()
model = Model(data)


def test():
    que_text = "孕妇检查四维彩超的时候医生会给家属进去看吗"
    ans_text = '一般家属不能进检查室的。孕期注意休息，均衡饮食，观察有无不适的症状，定期孕检。'
    candidate = model.predict(que_text=que_text)
    reference = list(jieba.lcut(ans_text))
    print(candidate)
    print(reference)
    score = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
    print(score)
    pass


def eval():
    train = pd.read_csv('data/input/dev.csv')
    cols = ['que_text']
    ans_text_index = list(train.columns).index('ans_text')
    x_train = train[cols]
    all_score = 0
    rows = []
    for i, row in x_train.iterrows():
        rows.append(row)
    candidates = model.predict_all(rows)
    for i, candidate in enumerate(candidates):
        reference = list(jieba.lcut(train.iloc[i, ans_text_index]))
        print(reference)
        print(candidate)
        all_score += sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
    print('score:', all_score / train.shape[0])


if __name__ == '__main__':
    data = Dataset()
    model = Model(data)
    test()
    # eval()
