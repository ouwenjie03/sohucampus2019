#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@license: NeteaseGame Licence 
@contact: ouwenjie@corp.netease.com
@file: format_data.py
@time: 2019/4/9 12:33


"""

import re
import json


def load_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line.strip())
            news_id = data['newsId']
            if 'coreEntityEmotions' in data:
                entities = data['coreEntityEmotions']
            else:
                entities = []
            title = data['title']
            content = data['content']

            sentences = [title]
            for seq in re.split(r'[\n。？！?!]', content):
                seq = re.sub(r'^[^\u4e00-\u9fa5A-Za-z0-9]+', '', seq)
                if len(seq) > 0:
                    sentences.append(seq)

            for seq in sentences:
                # find entities
                ent_spans = []
                ent_emotions = []
                for ent_emo in entities:
                    ent = ent_emo['entity']
                    emo = ent_emo['emotion']

                    last_idx = 0
                    while True:
                        if last_idx >= len(seq):
                            break
                        start = seq[last_idx:].find(ent)
                        if start == -1:
                            break
                        end = start + len(ent)
                        ent_spans.append((start + last_idx, end + last_idx))
                        ent_emotions.append(emo)
                        last_idx = end + last_idx

                sub_ent = []
                sub_emo = []
                sub_c = []
                for i, c in enumerate(seq):
                    sub_c.append(c)
                    ent_tag = 'O'
                    emo_tag = 'EMPTY'
                    for em, sp in zip(ent_emotions, ent_spans):
                        if sp[0] == i:
                            if sp[0] == sp[1]:
                                ent_tag = 'S'
                            else:
                                ent_tag = 'B'
                            emo_tag = em
                            break
                        elif sp[1]-1 == i:
                            ent_tag = 'E'
                            emo_tag = em
                            break
                        elif sp[0] < i < sp[1]-1:
                            ent_tag = 'I'
                            emo_tag = em
                            break
                    sub_ent.append(ent_tag)
                    sub_emo.append(emo_tag)

                for c, en, em in zip(seq, sub_ent, sub_emo):
                    print('{}\t{}\t{}'.format(c, en, em))
                print()


if __name__ == '__main__':
    import sys
    # data_file = './coreEntityEmotion_tiny.txt'
    data_file = sys.argv[1]
    load_data(data_file)