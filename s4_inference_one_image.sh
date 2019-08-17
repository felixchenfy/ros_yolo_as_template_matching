#!/bin/bash

# -- 0. Refresh yolo config files
python main_setup.py                 \
    --verify_mask           False    \
    --augment_imgs          False    \
    --setup_train_test_txt  False    \
    --setup_yolo            True   

# -- 1. Set image filename
image_filename="data/custom1_eval/00011.png"

# -- 2. Detect
python src/detect_one_image.py \
    --config_path "config/config.yaml" \
    --weights_path "weights/yolo_trained.pth" \
    --image_filename $image_filename
