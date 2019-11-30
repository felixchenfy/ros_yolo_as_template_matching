#!/bin/bash

# -- 0. Refresh yolo config files
python main_setup.py                 \
    --verify_mask           False    \
    --augment_imgs          False    \
    --setup_train_test_txt  False    \
    --setup_yolo            True   

# -- 1. Select one of the 3 data sources by commenting out the other two

# src_data_type="webcam"
# image_data_path="none"

# src_data_type="folder"
# image_data_path="test_data/images/"

src_data_type="video"
image_data_path="test_data/video.avi"

# -- 2. Detect
python src/detect_images.py \
    --config_path "config/config.yaml" \
    --weights_path "weights/yolo_trained.pth" \
    --src_data_type $src_data_type \
    --image_data_path $image_data_path