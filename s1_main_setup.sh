#!/bin/bash
python main_setup.py                 \
    --config_file config/config.yaml \
    --verify_mask           True     \
    --augment_imgs          True     \
    --setup_train_test_txt  True     \
    --setup_yolo            True    