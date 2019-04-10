#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@contact: https://github.com/ouwenjie03
@file: DataLoader_char.py
@time: 2019/4/9 11:08


"""

import numpy as np
import os
import pickle


class DataLoader_char:
    def __init__(self, config):
        self.config = config

        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.words = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

        self.entity2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, 'B': 4, 'I': 5, 'E': 6, 'S': 7, 'O': 8}
        self.entities = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'B', 'I', 'E', 'S', 'O']

        self.emotion2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, 'POS': 4, 'NEG': 5, 'NORM': 6, 'EMPTY': 7}
        self.emotions = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'POS', 'NEG', 'NORM', 'EMPTY']

        self.datas = []
        self.n_datas = -1

    def load_pkls(self):
        words_file = os.path.join(self.config.pkls_dir, 'words.pkl')
        word2idx_file = os.path.join(self.config.pkls_dir, 'word2idx.pkl')
        if os.path.exists(words_file):
            self.words = pickle.load(open(words_file, 'rb'))
            self.word2idx = pickle.load(open(word2idx_file, 'rb'))
        else:
            now_i = len(self.words)
            with open(self.config.train_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    tokens = line.strip().split('\t')
                    if len(tokens) < 3:
                        continue
                    c = tokens[0]
                    if c not in self.word2idx:
                        self.word2idx[c] = now_i
                        self.words.append(c)
                        now_i += 1
            pickle.dump(self.words, open(words_file, 'wb'))
            pickle.dump(self.word2idx, open(word2idx_file, 'wb'))

    def load_data(self, data_file):
        sub_seq = []
        sub_ent = []
        sub_emo = []
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                if len(tokens) < 3:
                    if len(sub_seq) > 0:
                        self.datas.append((sub_seq, sub_ent, sub_emo))
                    sub_seq = []
                    sub_ent = []
                    sub_emo = []
                    continue
                sub_seq.append(self.word2idx.get(tokens[0], self.word2idx['<UNK>']))
                sub_ent.append(self.entity2idx.get(tokens[1], self.entity2idx['<UNK>']))
                sub_emo.append(self.emotion2idx.get(tokens[2], self.emotion2idx['<UNK>']))

        self.n_datas = len(self.datas)

    def get_spaces(self):
        return len(self.words), len(self.entities), len(self.emotions)

    def get_batch(self, batch_size, is_shuffle=False):
        random_idxs = list(range(self.n_datas))
        if is_shuffle:
            np.random.shuffle(random_idxs)
        n_step = int(np.ceil(self.n_datas / batch_size))
        for i in range(n_step):
            start = i * batch_size
            end = start + batch_size
            batch_data = [self.datas[_i] for _i in random_idxs[start:end]]

            yield batch_data

    def format_predict(self, seqs):
        seq_idxs = []
        for seq in seqs:
            tmp_sidx = []
            for c in seq:
                tmp_sidx.append(self.word2idx.get(c, self.word2idx['<UNK>']))
            seq_idxs.append(tmp_sidx)
        seq_lens = [len(_x) for _x in seq_idxs]
        return seq_idxs, seq_lens


if __name__ == '__main__':
    from src.Config import Config
    config = Config()
    config.pkls_dir = '../pkls'
    config.train_file = '../data/valid.bie'

    dl = DataLoader_char(config)
    dl.load_pkls()

    dl.load_data(config.train_file)

    for data in dl.get_batch(10):
        seqs = [_x[0] for _x in data]
        ents = [_x[1] for _x in data]
        emos = [_x[2] for _x in data]
        for seq, ent, emo in zip(seqs, ents, emos):
            print([dl.words[_x] for _x in seq])
            print([dl.entities[_x] for _x in ent])
            print([dl.emotions[_x] for _x in emo])
        break
