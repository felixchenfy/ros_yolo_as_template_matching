#!/bin/bash

# -- 1. Move trained weight file to weights/yolo_trained.ckpt
mv checkpoints/yolov3_ckpt_100.pth weights/yolo_trained.pth 2>/dev/null || :

# -- 2. Setup yolo config files
python main_setup.py                 \
    --verify_mask           False    \
    --augment_imgs          False    \
    --setup_train_test_txt  False    \
    --setup_yolo            True   

# -- 3. Select one of the 3 data sources by commenting out the other two

# data_source="webcam"
# image_data_path="none"

# data_source="folder"
# image_data_path="data/custom1_eval/"
# # image_data_path="data/custom1_generated/valid_images/"

data_source="video"
image_data_path="data/custom1_eval/video.avi"

# -- 4. Start detecting

WEIGHTS_PATH="weights/yolo_trained.pth"
python src/detect_images.py \
    --weights_path $WEIGHTS_PATH \
    --conf_thres 0.99 \
    --nms_thres 0.3 \
    --batch_size 1 \
    --data_source $data_source \
    --image_data_path $image_data_path
