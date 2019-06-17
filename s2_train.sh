#!/bin/bash
python src/train.py \
    --epochs 100 \
    --learning_rate 0.001 \
    --checkpoint_interval 10 \
    --pretrained_weights weights/darknet53.conv.74 \
    --batch_size 4 
