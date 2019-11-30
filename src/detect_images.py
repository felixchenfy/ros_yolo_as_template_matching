# -*- coding: future_fstrings -*-
from __future__ import division

'''
Detect images by YOLOv3. 
The images can be read from either:
    (1) Web camera.
    (2) A folder of images.
    (3) A video file.
'''

import os
import sys
import time
import datetime
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if 1:  # Set path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__)) + \
        "/../"  # root of the project
    sys.path.append(ROOT)
    from config.config import read_all_args

    import src.lib_yolo_detect as lib_yolo_detect
    import utils.lib_plot as lib_plot
    from utils.lib_yolo_plot import Yolo_Detection_Plotter_CV2, Yolo_Detection_Plotter_PLT


def set_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str,
                        default=ROOT + "config/config.yaml",
                        help="path to config file")
    parser.add_argument("-w", "--weights_path", type=str,
                        default="weights/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("-t", "--src_data_type",
                        choices=['folder', 'video', 'webcam'],
                        required=False,
                        type=str,
                        default="webcam",
                        help="read data from a folder, video file, of webcam")
    parser.add_argument("-i", "--image_data_path", type=str,
                        required=False,
                        default="none",
                        help="depend on '--src_data_type', set this as: a folder, or a video file,")
    parser.add_argument("-o", "--output_folder", type=str,
                        required=False,
                        default=ROOT + "output/",
                        help="Detection result images will be saved here.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Set parameters
    args = set_inputs()
    IF_SHOW = True  # if false, draw the image and save to file, but not show out
    IF_SINGLE_INSTANCE = False  # single instance for each class

    # Save result data to this folder
    OUTPUT_FOLDER = args.output_folder
    print("Result images are saved to: " + OUTPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Init
    detector = lib_yolo_detect.ObjDetector(args.config_path, args.weights_path)
    args_inference = detector.args_inference
    dataloader = lib_yolo_detect.set_dataloader(args.src_data_type, args.image_data_path,
                                                args_inference.img_size, args_inference.batch_size, args_inference.n_cpu)

    # ------------------------- Start loop through images to detect ---------------

    print("\nPerforming object detection:")
    prev_time = time.time()
    cnt_img = 0

    for batch_i, (imgs_path, imgs) in enumerate(dataloader):

        # -- Detect
        # Argument:
        #   imgs: shape [B, W, H, 3], tensor, rgb
        # Return:
        #   imgs_detections: A list of `detections`. Nx7.
        #                    `detections`: [bbox, conf, cls_conf, cls_idx]
        #                                  bbox = [x1, y1, x2, y2] in the image coordinate
        imgs_detections = detector.detect_torch_imgs(imgs)

        # -- Print progress.
        if 1:
            current_time = time.time()
            inference_time = datetime.timedelta(
                seconds=current_time - prev_time)
            prev_time = current_time
            print("\nBatch %d: %d images; Inference Time: %s" %
                  (batch_i, len(imgs), inference_time))

        # -- Draw "detections" onto each image
        for img_i, (img, path, detections) in enumerate(zip(imgs, imgs_path, imgs_detections)):

            img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)  # torch -> cv2

            # Plot
            img_disp = detector.draw_bboxes(img, detections)
            cv2.imshow("image", img_disp)
            cv2.waitKey(10)

            # Save
            filename = "{}/{}".format((OUTPUT_FOLDER),
                                      (os.path.basename(path)))
            cv2.imwrite(filename, img_disp)

    print("Result images are saved to: " + OUTPUT_FOLDER)
    cv2.destroyAllWindows()
        