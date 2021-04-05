#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (DEQ-TrellisNet)...'
    python train_trellisnet.py \
        --data ../data/penn/ \
        --dataset ptb \
        --n_layer 58 \
        --d_embed 400 \
        --nhid 1000 \
        --nout 400 \
        --epochs 500 \
        --optim SGD \
        --lr 20 \
        --clip 0.225 \
        --batch_size 16 \
        --seq_len 110 \
        --subseq_len 110 \
        --dropout 0.45 \
        --dropouti 0.45 \
        --wdrop 0.5 \
        --emb_dropout 0.1 \
        --dropouth 0.28 \
        --weight_decay 1.2e-6 \
        --wnorm \
        --seed 1111 \
        --cuda \
        --anneal 10 \
        --log-interval 100 \
        --when -1 \
        --ksize 2 \
        --dilation 1 \
        --n_experts 0 \
        --multi_gpu \
        --f_thresh 55 \
        --b_thresh 80 \
        --gpu0_bsz -1 \
        --pretrain_steps 30000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Not supported yet'
else
    echo 'unknown argment 1'
fi
