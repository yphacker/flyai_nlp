# coding=utf-8
# author=yphacker

import re
import jieba
import pandas as pd


def clean_str(string):
    """
    该函数的作用是去掉一个字符串中的所有非中文字符
    :param string:
    :return: 返回处理后的字符串
    """
    string.strip('\n')
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def cut_line(line):
    """
    该函数的作用是 先清洗字符串，然后分词
    :param line:
    :return: 分词后的结果，如 ：     衣带  渐宽  终  不悔
    """
    line = clean_str(line)
    seg_list = jieba.cut(line)
    return seg_list


df = pd.read_csv('data/input/dev.csv')
texta = list(df['texta'])
textb = list(df['textb'])
max_len = 0
for i in range(len(texta)):
    text = []
    text.extend(cut_line(texta[i]))
    text.extend(cut_line(texta[i]))
    max_len = max(max_len, len(text))
print(max_len)
