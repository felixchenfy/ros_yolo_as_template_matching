# -*- coding: future_fstrings -*-
from __future__ import division

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    from config.config import read_all_args
    import warnings
    warnings.filterwarnings("ignore")
    
import os
import sys
import time
import datetime
import argparse
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import types

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import utils.lib_yolo_funcs as yolo
from utils.lib_common_funcs import Timer
from utils.lib_yolo_plot import Yolo_Detection_Plotter_by_cv2, Yolo_Detection_Plotter_by_very_slow_plt
from utils.lib_plot import show 

# ===========================================================================

def set_arguments():
    
    # Set default args
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    
    # Set which image(s) to detect
    parser.add_argument("--image_filename", type=str,
                        help="filename of the image")
    args = parser.parse_args()
    
    # Set model path from config/config.yaml
    configs = read_all_args("config/config.yaml")
    args.model_def = configs.f_yolo_config 
    args.data_config = configs.f_yolo_data 
    args.class_path = configs.f_yolo_classes 

    # Return
    return args 


class Detector(object):
    ''' Yolo detector for single image '''
    def __init__(self, args):
        self.model = yolo.create_model(args)
        self.classes = yolo.load_classes(args.class_path)  # Extracts class labels from file
        self.plotter = Yolo_Detection_Plotter_by_cv2(IF_SHOW=True, 
                                                cv2_waitKey_time=0, resize_scale=1.5)
        self.args = args 
        
    def detect(self, 
               img,
               IF_SINGLE_INSTANCE = False, # single instance for each class
               
        ):
        # Change format to the required one
        imgs = self._cv2_format_to_detector(img) 
    
        # Detect
        imgs_detections = yolo.detect_targets(
            self.args, self.model, imgs, IF_SINGLE_INSTANCE)
        
        # Return
        detections = imgs_detections[0] # there is only 1 image here
        return detections 
    
    def plot(self, img, detections):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_disp = self.plotter.plot(img, detections, self.classes)
        return img_disp
    
    def _cv2_format_to_detector(self, img): # Output: [1, W, H, 3], tensor, rgb
        ''' Change image format to what the detector requires. '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = torch.from_numpy(img).unsqueeze(0)
        return imgs 
    
if __name__ == "__main__":
    
    # Load image
    args = set_arguments()
    img = cv2.imread(args.image_filename, cv2.IMREAD_COLOR)
    
    # Detect
    detector = Detector(args)
    detections = detector.detect(img)
    
    # Plot
    img_disp = detector.plot(img, detections)
    
    # Save result
    OUTPUT_FOLDER = "output/"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    cv2.imwrite(OUTPUT_FOLDER + os.path.basename(args.image_filename), img=img_disp)
    