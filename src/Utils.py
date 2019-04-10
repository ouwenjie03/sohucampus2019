#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@contact: https://github.com/ouwenjie03
@file: Utils.py
@time: 2019/4/9 16:45


"""

import torch
import torch.nn as nn


def weight_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.LSTM):
        # 注意单向或者双向
        for part in m.all_weights:
            for w in part:
                if len(w.size()) >= 2:
                    # weight
                    nn.init.xavier_uniform_(w)
                else:
                    # bias
                    w.data.fill_(0.)
                    n = w.size(0)
                    # forget gate
                    start, end = n//4, n//2
                    w.data[start:end].fill_(1.)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


def make_mask(lengths, max_len, pad_value=True, value=None):
    masks = []
    for l in lengths:
        masks.append([(not pad_value) for _ in range(l)] + [pad_value for _ in range(max_len-l)])
    if value is not None:
        masks = torch.FloatTensor(masks)
        masks *= value
    else:
        masks = torch.ByteTensor(masks)
    return masks


def accuracy(labels, predicts, ignore_idxs):
    assert len(labels) == len(predicts)
    cnt = 0.0
    total_cnt = 0.0
    for l, p in zip(labels, predicts):
        if l == 0:
            continue
        if l in ignore_idxs and p in ignore_idxs:
            continue
        if l == p:
            cnt += 1.0
        total_cnt += 1.0
    return cnt, total_cnt

