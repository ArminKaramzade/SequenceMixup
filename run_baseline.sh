#!/bin/bash

# ALPHA in [0.1, 0.2, 0.5, 1, 2]

CELL=LSTM
MODEL=PreOutputMixup
BPATH=./trained_models/baseline-${CELL}-${MODEL}
DEVICE=cuda:0

# while [[ $# -gt 0 ]]
# do
# key="$1"

# case $key in
#     -p|--path)
#     BPATH="$2"
#     shift
#     shift
#     ;;
#     -m|--model)
#     MODEL="$2"
#     shift
#     shift
#     ;;
#     -c|--cell)
#     CELL="$2"
#     shift
#     shift
#     ;;
#     -d|--device)
#     DEVICE="$2"
#     shift
#     shift
#     ;;
#     -a|--alpha)
#     ALPHA="$2"
#     shift
#     shift
#     ;;
#     -b|--bandwidth-percentage)
#     BP="$2"
#     shift
#     shift
#     ;;
#     *)
#     shift
#     ;;
# esac
# done

i=1;
while [[ $i -lt 5 ]]
do
    for ALPHA in 0.1 0.2 0.5 1 2
    do
        python main.py --base-path ${BPATH}-${i}-ALPHA=${ALPHA} --model ${MODEL} --cell-type ${CELL} --device ${DEVICE} \
                    --epochs 50 --embeddings glove --train-with-dev --monitor-test \
                    --hidden-size 256 --word-dropout 0 --locked-dropout 0 --relearn-embeddings \
                    --lambdas-generator beta --alpha ${ALPHA} --beta ${ALPHA}\
                    --data-path data/conll_03 --tag-name ner\
                    --patience 0 --step-size 10

    done
    i=$(($i+1))
done
