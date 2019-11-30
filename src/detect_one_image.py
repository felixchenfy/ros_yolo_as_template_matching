# -*- coding: future_fstrings -*-
from __future__ import division

'''
Detect one image by YOLOv3. 
'''

import os
import sys
import datetime
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

if 1:  # Set path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__)) + \
        "/../"  # root of the project
    sys.path.append(ROOT)
    from config.config import read_all_args
    import warnings
    warnings.filterwarnings("ignore")

    import src.lib_yolo_detect as lib_yolo_detect
    import utils.lib_plot as lib_plot
    from utils.lib_yolo_plot import Yolo_Detection_Plotter_CV2, Yolo_Detection_Plotter_PLT

# ===========================================================================


def set_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_filename", type=str,
                        default=ROOT + "test_data/images/00014.png",
                        help="filename of the image")
    parser.add_argument("-w", "--weights_path", type=str,
                        default=ROOT + "weights/yolo_trained.pth",
                        help="path to weights file")
    parser.add_argument("-c", "--config_path", type=str,
                        default=ROOT + "config/config.yaml",
                        help="path to config file")
    parser.add_argument("-o", "--output_folder", type=str,
                        default=ROOT + "output/",
                        help="Detection result images will be saved here.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Load image.
    args = set_inputs()
    img = cv2.imread(args.image_filename, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Invalid image: " + args.image_filename)

    # Init detector.
    detector = lib_yolo_detect.ObjDetector(args.config_path, args.weights_path)

    # Detect.
    # 2d list. Each list contains info of one object.
    detections = detector.detect_cv2_img(img)

    img_disp = detector.draw_bboxes(img, detections)

    # Test detecting multiple images.
    TEST_DETECT_IMGS = False
    if TEST_DETECT_IMGS:
        imgs = [img, img, img]  # duplicate image
        imgs_detections = detector.detect_cv2_imgs(imgs)
        img_disps = [detector.draw_bboxes(img, detections)
                     for detections in imgs_detections]
        img_disp = np.hstack(img_disps)

    # Plot.
    lib_plot.cv2_imshow(img_disp, time_ms=0)

    # Save result.
    OUTPUT_FOLDER = args.output_folder
    print("Result images are saved to: " + OUTPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    dst_filename = OUTPUT_FOLDER + os.path.basename(args.image_filename)
    cv2.imwrite(dst_filename, img=img_disp)
    