#!/bin/bash

CELL=LSTM
# MODEL=PreOutputMixup
# MODEL=InputMixup
MODEL=ThroughTimeMixup
ALPHA=0.5
BPATH=./trained_models/baseline-${CELL}-${MODEL}
DEVICE=cuda:0


i=1;
while [[ $i -lt 5 ]]
do
    for RHO in 1
    do
        python main.py --base-path ${BPATH}-${i}-ALPHA=${ALPHA}-RHO-${RHO} --model ${MODEL} --cell-type ${CELL}\
                    --device ${DEVICE} \
                    --epochs 50 --embeddings glove --train-with-dev --monitor-test \
                    --hidden-size 256 --word-dropout 0 --locked-dropout 0 --relearn-embeddings \
                    --lambdas-generator beta --alpha ${ALPHA} --beta ${ALPHA} --rho ${RHO}\
                    --data-path data/conll_03 --tag-name ner\
                    --patience 0 --step-size 10
    done
    i=$(($i+1))
done