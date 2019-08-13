#!/bin/bash
python src/train.py \
    --epochs 1000 \
    --learning_rate 0.0005 \
    --checkpoint_interval 20 \
    --pretrained_weights weights/darknet53.conv.74 \
    --batch_size 2 
