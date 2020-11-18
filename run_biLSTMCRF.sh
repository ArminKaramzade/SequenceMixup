#!/bin/bash

python main.py --base-path ./trained_models/model_seleection/im-alpha-1 \
               --model InputMixup \
               --device cuda:0 \
               --cell-type LSTM \
               --embeddings-storage-mode none\
               --epochs 250 --embeddings glove+pooled --monitor-dev \
               --bidirectional --use-crf \
               --hidden-size 256 --word-dropout 0 --locked-dropout 0.5 --relearn-embeddings \
               --lambdas-generator beta --alpha 1 --beta 1 --rho 0 \
               --data-path data/conll_03 --tag-name ner \
               --patience 6