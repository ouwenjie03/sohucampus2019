#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@contact: https://github.com/ouwenjie03
@file: main_softmax.py
@time: 2019/4/9 16:58


"""

from src.DataLoader_char import DataLoader_char as DataLoader
from src.Config import Config
from src.BiLSTM import BiLSTM

import time
import os
import torch
import json
import re
import logging
from tensorboardX import SummaryWriter

# logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s')


def train(config, save_dir):
    logging.basicConfig(
        handlers=[logging.StreamHandler(), logging.FileHandler("./{}_train.log".format(config.model_name))],
        level=logging.INFO,
        format="[%(asctime)s] : %(message)s",
    )

    writer = SummaryWriter('{}_log'.format(config.model_name))

    logging.info('train dataset loading...')
    stime = time.time()
    train_dl = DataLoader(config)
    train_dl.load_pkls()
    train_dl.load_data(config.train_file)
    etime = time.time()
    logging.info('load train dataset cost: {}s'.format(etime-stime))

    logging.info('valid dataset loading...')
    stime = time.time()
    valid_dl = DataLoader(config)
    valid_dl.load_pkls()
    valid_dl.load_data(config.valid_file)
    etime = time.time()
    logging.info('load train dataset cost: {}s'.format(etime - stime))

    n_words, n_entities, n_emotions = train_dl.get_spaces()

    device = torch.device('cuda' if config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = BiLSTM(device, config, n_words, n_entities, n_emotions).to(device)

    optim = torch.optim.Adam(model.parameters(), config.lr)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, config.lr_decay)

    batch_i = 0
    valid_i = 0
    for epoch in range(config.epoch_size):
        for batch_datas in train_dl.get_batch(config.batch_size, is_shuffle=True):
            model.train()
            optim.zero_grad()

            stime = time.time()
            batch_i += 1

            seqs = [_x[0] for _x in batch_datas]
            seq_lens = [len(_x) for _x in seqs]
            ents = [_x[1] for _x in batch_datas]
            emos = [_x[2] for _x in batch_datas]

            entity_scores, emotion_scores = model(seqs, seq_lens)
            entity_loss = model.get_loss(entity_scores, ents, seq_lens)
            emotion_loss = model.get_loss(emotion_scores, emos, seq_lens)

            loss = entity_loss + emotion_loss

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
            optim.step()

            etime = time.time()
            writer.add_scalar('train_loss', loss.item(), batch_i)
            writer.add_scalar('train_entity_loss', entity_loss.item(), batch_i)
            writer.add_scalar('train_emotion_loss', emotion_loss.item(), batch_i)
            logging.info('TRAIN | epoch {} | batch {} | loss {} | cost {:.5f}s'.format(epoch, batch_i, loss.item(), etime-stime))
            torch.cuda.empty_cache()

        model.eval()
        valid_entity_losses = 0.0
        valid_emotion_losses = 0.0
        li = 0
        # valid
        with torch.no_grad():
            for valid_batch_datas in valid_dl.get_batch(config.batch_size):
                valid_seqs = [_x[0] for _x in valid_batch_datas]
                valid_seq_lens = [len(_x) for _x in valid_seqs]
                valid_ents = [_x[1] for _x in valid_batch_datas]
                valid_emos = [_x[2] for _x in valid_batch_datas]

                valid_entity_scores, valid_emotion_scores = model(valid_seqs, valid_seq_lens)
                valid_entity_loss = model.get_loss(valid_entity_scores, valid_ents, valid_seq_lens)
                valid_emotion_loss = model.get_loss(valid_emotion_scores, valid_emos, valid_seq_lens)

                entity_predicts = torch.argmax(valid_entity_scores, dim=-1)
                emotion_predicts = torch.argmax(valid_emotion_scores, dim=-1)
                if entity_predicts.is_cuda:
                    entity_predicts = entity_predicts.to(torch.device('cpu'))
                    emotion_predicts = emotion_predicts.to(torch.device('cpu'))
                entity_predicts = entity_predicts.numpy()
                emotion_predicts = emotion_predicts.numpy()
                ent_preds = []
                emo_preds = []
                for i, l in enumerate(valid_seq_lens):
                    ent_preds.append(entity_predicts[i, :l])
                    emo_preds.append(emotion_predicts[i, :l])

                valid_entity_losses += valid_entity_loss.item() * len(valid_ents)
                valid_emotion_losses += valid_emotion_loss.item() * len(valid_emos)
                li += len(valid_emos)

                torch.cuda.empty_cache()

            writer.add_scalar('valid_entity_loss', valid_entity_losses/li, valid_i)
            writer.add_scalar('valid_emotion_loss', valid_emotion_losses/li, valid_i)

        save_path = '{}/{}_{}.pkl'.format(save_dir, config.model_name, valid_i)
        model.save_model(save_path)

        valid_i += 1


def predict(config, load_file, test_file):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] : %(message)s",
    )

    logging.info('config loading...')
    stime = time.time()
    train_dl = DataLoader(config)
    train_dl.load_pkls()

    n_words, n_entities, n_emotions = train_dl.get_spaces()
    device = torch.device('cuda' if config.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = BiLSTM(device, config, n_words, n_entities, n_emotions)
    model.load_model(load_file)
    model = model.to(device)
    model.eval()
    etime = time.time()
    logging.info('load config cost: {}s'.format(etime - stime))

    with open(test_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line.strip())
            news_id = data['newsId']
            title = data['title']
            content = data['content']
            sentences = [title]
            for seq in re.split(r'[\n。？！?!]', content):
                seq = re.sub(r'^[^\u4e00-\u9fa5A-Za-z0-9]+', '', seq)
                if len(seq) > 0:
                    sentences.append(seq)

            with torch.no_grad():
                seq_idxs, seq_lens = train_dl.format_predict(sentences)
                entity_scores, emotion_scores = model(seq_idxs, seq_lens)

                entity_predicts = torch.argmax(entity_scores, dim=-1)
                emotion_predicts = torch.argmax(emotion_scores, dim=-1)

            ent2emo = {}
            for seq, ent_ps, emo_ps in zip(sentences, entity_predicts, emotion_predicts):
                start = -1
                end = -1
                for i, t in enumerate(ent_ps):
                    t = t.item()
                    if 'B' == train_dl.entities[t]:
                        start = i
                    elif 'E' == train_dl.entities[t]:
                        end = i
                        if end >= 0 and start >= 0 and (end+1-start) < 11:
                            entity = seq[start:end+1]
                            emotion = 'NORM'
                            for emop in emo_ps[start:end]:
                                if emop == train_dl.emotion2idx['POS']:
                                    emotion = 'POS'
                                elif emop == train_dl.emotion2idx['NEG']:
                                    emotion = 'NEG'
                            if ent2emo.get(entity, 'NORM') == 'NORM':
                                ent2emo[entity] = emotion

                            start = -1
                            end = -1
                    elif 'I' == train_dl.entities[t]:
                        end = i
                    elif train_dl.entities[t] == 'O':
                        if end >= 0 and start >= 0 and (end+1-start) < 11:
                            entity = seq[start:end+1]
                            emotion = 'NORM'
                            for emop in emo_ps[start:end]:
                                if emop == train_dl.emotion2idx['POS']:
                                    emotion = 'POS'
                                elif emop == train_dl.emotion2idx['NEG']:
                                    emotion = 'NEG'
                            if ent2emo.get(entity, 'NORM') == 'NORM':
                                ent2emo[entity] = emotion

                            start = -1
                            end = -1

                if end >= 0 and start >= 0 and (end + 1 - start) < 11:
                    entity = seq[start:end + 1]
                    emotion = 'NORM'
                    for emop in emo_ps[start:end]:
                        if emop == train_dl.emotion2idx['POS']:
                            emotion = 'POS'
                        elif emop == train_dl.emotion2idx['NEG']:
                            emotion = 'NEG'
                    if ent2emo.get(entity, 'NORM') == 'NORM':
                        ent2emo[entity] = emotion

            for k in ent2emo:
                for other_k in ent2emo.keys():
                    if k == other_k:
                        continue
                    if k in other_k:
                        emo = ent2emo[k]
                        if ent2emo[other_k] == 'NORM':
                            ent2emo[other_k] = emo
                        ent2emo[k] = 'DELETE'
            # print(ent2emo)

            ents = []
            emos = []
            for k, v in ent2emo.items():
                if v != 'DELETE':
                    ents.append(k.strip())
                    emos.append(v)

            print('{}\t{}\t{}'.format(news_id, ','.join(ents), ','.join(emos)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='python main_softmax.py ' +
                    '--do [train/test] ' +
                    '--model BiLSTM ' +
                    '--tag_type [bie/bi/ie]' +
                    '--n_gram n' +
                    '--path /path/to/save(load)'
    )
    parser.add_argument('--do', type=str, default='train')
    parser.add_argument('--model', type=str, default='BiLSTM')
    parser.add_argument('--path', type=str, default='save_models/')
    args = parser.parse_args()

    config = Config()
    config.model_name = args.model

    if 'train' in args.do:
        if not os.path.isdir(args.path):
            os.mkdir(args.path)
        train(config, args.path)
    elif 'predict' in args.do:
        predict(config, args.path, './data/coreEntityEmotion_test_stage1.txt')
