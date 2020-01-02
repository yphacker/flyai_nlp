# coding=utf-8
# author=yphacker

import argparse
import numpy as np
import tensorflow as tf
from flyai.dataset import Dataset
import config
from config import MODEL_PATH
from model import Model
from baseline_model import NerNet
from utils import load_word2vec_embedding

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
parser.add_argument("-e", "--EPOCHS", default=32, type=int, help="train epochs")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(batch=args.BATCH, epochs=args.EPOCHS)
# 模型操作辅助类
modelpp = Model(dataset)

# 训练神经网络
config.batch_size = args.BATCH
embedding = load_word2vec_embedding(config.vocab_size)


def train():
    model = NerNet(embedding, args.BATCH)
    model.train(dataset, modelpp)


train()
