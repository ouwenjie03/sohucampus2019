#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@contact: https://github.com/ouwenjie03
@file: Config.py
@time: 2019/4/9 16:37


"""


class Config:
    def __init__(self):
        self.pkls_dir = './pkls'
        self.train_file = './data/train.bie'
        self.valid_file = './data/valid.bie'

        self.lr = 1e-3
        self.lr_decay = 1.

        self.model_name = 'BiLSTM'
        self.device = 'cuda'

        self.dropout_rate = 0.5

        # for BiLSTM-NNCRF
        self.word_embed_dim = 100
        self.label_embed_dim = 100
        self.max_sent_len = 200

        # for Char
        self.char_embed_dim = 500
        # for Char_CNN   暂时不用
        self.char_tone_embed_dim = 64
        self.char_cnn_kernel_size = 5

        self.batch_size = 256
        self.epoch_size = 5

        self.clip_grad = 10.
