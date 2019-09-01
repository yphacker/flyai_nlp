# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

num_dims = 100


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input shape: 词向量维度，hidden个数，lstm层数
        self.LSTM_stack = nn.LSTM(num_dims, 64, num_layers=2, batch_first=True, bidirectional=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.relu1 = nn.ReLU(True)
        self.fc1 = nn.Linear(30 * 64 * 2, 128)  ##  (max sentence length * hidden layer, 512)
        self.relu2 = nn.ReLU(True)
        self.dp = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x1, x2):
        x1, _ = self.LSTM_stack(x1.float())  # (batch, sentence_len, hidden_units)
        x2, _ = self.LSTM_stack(x2.float())
        x = x1 * x2

        # use every word in the sentence
        x = x.contiguous().view(-1, x.size(1) * x.size(2))
        x = self.relu1(x)
        x = self.fc1(x.float())
        x = self.relu2(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = x / torch.norm(x)
        return x
