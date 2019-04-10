#! /bin/sh
#
# run.sh
# Copyright (C) 2019 ouwenjie <https://github.com/ouwenjie03>
#
# Distributed under terms of the MIT license.
#


mkdir ./pkls
mkdir ./models

python ./data/format_data.py ./data/coreEntityEmotion_train.txt > ./data/train.bie
python ./data/format_data.py ./data/coreEntityEmotion_example.txt > ./data/valid.bie


CUDA_VISIBLE_DEVICES=0 python ./main_softmax.py --do train --path ./models

CUDA_VISIBLE_DEVICES=0 python ./main_softmax.py --do predict --path ./models/BiLSTM_4.pkl > result.txt
