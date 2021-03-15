#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (DEQ-TrellisNet)...'
    python train_trellisnet.py \
        --cuda \
        --data ../data/penn/ \
        --dataset ptb \
        --n_layer 40 \
        --d_embed 400\
        --nhid 1000 \
        --nout 400 \
        --clip 0.007 \
        --dropout 0.45 \
        --dropouti 0.45 \
        --wdrop 0.5 \
        --emb_dropout 0.1 \
        --dropouth 0.28 \
        --weight_decay 1.2e-6 \
        --optim Adam \
        --lr 1e-3 \
        --anneal 5 \
        --pretrain_steps 30000 \
        --seq_len 110 \
        --subseq_len 55 \
        --f_thres 45 \
        --b_thres 45 \
        --batch_size 20 \
        --gpu0_bsz 7 \
        --multi_gpu \
        --use_gpus 0 1 2 3 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Not supported yet'
else
    echo 'unknown argment 1'
fi
