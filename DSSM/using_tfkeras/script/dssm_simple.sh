#!/bin/sh 
# 训练dssm
echo "execute $BASH_SOURCE"
cd "`dirname $0`/.."
pwd

PYBIN="python3"

#MODE="TRAIN"
#MODE="EVAL"
#MODE="PREDICT"
#MODE="QUERY_EMB"
#MODE="DOC_EMB"


MODE="TRAIN"
${PYBIN} main.py                          \
    --name "DSSM"                         \
    --mode "$MODE"                        \
    --train "./data/train/dssm/"          \
    --test "./data/test/dssm/"            \
    --model "./data/model/keras/"         \
    --checkpoint "./data/checkpoint/"     \
    --tensorboard "./data/tensorboard"    
