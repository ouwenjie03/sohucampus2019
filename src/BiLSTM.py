#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@contact: https://github.com/ouwenjie03
@file: BiLSTM.py
@time: 2019/4/9 9:48


"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Utils import weight_init


class BiLSTM(nn.Module):
    def __init__(self, device, config, n_words, n_entities, n_emotions):
        super(BiLSTM, self).__init__()

        self.device = device
        self.config = config

        # dropout layer
        self.dropout = torch.nn.Dropout(config.dropout_rate)

        self.word_embed_dim = config.word_embed_dim
        self.label_embed_dim = config.label_embed_dim
        self.n_words = n_words
        self.n_entities = n_entities
        self.n_emotions = n_emotions

        # Embedding
        self.word_embeddings = nn.Embedding(self.n_words, self.word_embed_dim, padding_idx=0)

        # word layers
        self.word_in_dim = self.word_embed_dim
        self.word_bilstm = nn.LSTM(input_size=self.word_in_dim, hidden_size=self.word_embed_dim, num_layers=1, bidirectional=True)

        self.word_hidden_dim = self.word_embed_dim * 2
        self.lstm_out_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.word_hidden_dim, self.word_hidden_dim),
        )

        self.entity_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.word_hidden_dim, self.n_entities),
        )

        self.emotion_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.word_hidden_dim, self.n_emotions),
        )

        self.apply(weight_init)

    def forward(self, xs, xlens):
        batch_size = len(xs)

        # sort of len by decreasing order

        dec_idx = np.argsort(xlens)[::-1]
        sorted_len = sorted(xlens)[::-1]

        max_len = max(sorted_len)
        sorted_xs = []
        for i in dec_idx:
            sorted_xs.append(xs[i] + [0] * (max_len - len(xs[i])))
        # B*L*E
        sorted_xs = torch.LongTensor(sorted_xs).to(self.device)

        # B*L*E
        word_embs = self.word_embeddings(sorted_xs)
        word_embs = self.dropout(word_embs)

        # bilstm layer
        word_embs = word_embs.transpose(1, 0)
        pad_embs = nn.utils.rnn.pack_padded_sequence(word_embs, sorted_len)
        hidden_states, (hidden_n, cell_n) = self.word_bilstm(pad_embs)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states)

        # concat
        hidden_states = hidden_states.transpose(1, 0).contiguous().view(batch_size, max_len, self.word_embed_dim*2)
        word_features = self.lstm_out_linear(hidden_states)

        entity_scores = self.entity_linear(word_features)
        emotion_scores = self.emotion_linear(word_features)

        # reverse sort
        re_sid = np.argsort(dec_idx)
        # B*T*L
        entity_scores = entity_scores[re_sid]
        emotion_scores = emotion_scores[re_sid]

        return entity_scores, emotion_scores

    def get_loss(self, scores, targets, x_lens):
        # pad label
        probs = []
        labels = []
        for i in range(len(x_lens)):
            for j in range(x_lens[i]):
                probs.append(scores[i, j])
                labels.append(targets[i][j])

        probs = torch.stack(probs, dim=0)
        labels = torch.LongTensor(labels).to(self.device)

        loss = F.cross_entropy(probs, labels)

        return loss

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        model_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(model_dict)
