#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: future_fstrings -*-
from __future__ import division

'''
ROS server for YOLO:
Input/output from/to ROS topics.

Input: image.

Output: (1) image with detection results.
        (2) Detection results. (See `msg/DetectionResults.msg`.)
'''

# -- Torch
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
import torch

# -- ROS
import rospy

# Output message type.
from ros_yolo_as_template_matching.msg import DetectionResults

# -- Commons
import cv2
import numpy as np
import argparse
import yaml
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
import warnings
warnings.filterwarnings("ignore")

# -- My libraries
if True:
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    sys.path.append(ROOT)

    # Commons
    from utils.lib_common_funcs import Timer, SimpleNamespace
    import utils.lib_plot as lib_plot

    # ROS
    from utils.lib_ros_rgbd_pub_and_sub import ColorImageSubscriber, ColorImagePublisher

    # YOLO
    from config.config import read_all_args
    import src.lib_yolo_detect as lib_yolo_detect
    from utils.lib_yolo_plot import Yolo_Detection_Plotter_CV2, Yolo_Detection_Plotter_PLT


# ===========================================================================

def set_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights_path", type=str,
                        default=ROOT+"weights/yolo_trained.pth",
                        help="path to weights file")
    parser.add_argument("-c", "--config_path", type=str,
                        default=ROOT+"config/config.yaml",
                        help="path to config file")
    parser.add_argument("-i", "--src_topic_img", type=str,
                        help="ROS topic: src image for object detection.")
    parser.add_argument("-o", "--dst_topic_img", type=str,
                        help="ROS topic: dst image that shows the results.")
    parser.add_argument("-t", "--dst_topic_res", type=str,
                        help="ROS topic: dst results of each object's label, confidence and bbox.")
    args = parser.parse_args()
    return args


class DetectionResultsPublisher(object):
    def __init__(self, topic_name, queue_size=10):
        self._pub = rospy.Publisher(
            topic_name, DetectionResults, queue_size=queue_size)

    def publish(self, detections):
        '''
        Arguments:
            detections {list of PlaneParam}
        '''
        int32 N                 # Number of detected objects.


string label            # Nx1. Label of each object.
float32[] confidence    # Nx1. Confidence of each object. Range=[0, 1].
float32[] bbox          # Nx4. (x0, y0, x1, y1). Range=[0, 1].

 res = DetectionResults()
  res.N = len(plane_params)
   for pp in plane_params:
        res.norms.extend(pp.w.tolist())
        res.center_3d.extend(pp.pts_3d_center.tolist())
        res.center_2d.extend(pp.pts_2d_center.tolist())
        res.mask_color.extend(pp.mask_color.tolist())
    self._pub.publish(res)
    return


def main(args):

    # -- Input ROS topic.
    sub_img = ColorImageSubscriber(args.src_topic_img)

    # -- Output ROS topics.
    pub_img = ColorImagePublisher(args.dst_topic_img)

    img = cv2.imread(args.image_filename, cv2.IMREAD_COLOR)

    # Init detector
    detector = lib_yolo_detect.ObjDetector(args.config_path, args.weights_path)

    # Detect
    detections = detector.detect_cv2_img(img)
    img_disp = detector.draw_bboxes(img, detections)

    TEST_DETECT_IMGS = False
    if TEST_DETECT_IMGS:
        imgs = [img, img, img]  # duplicate image
        imgs_detections = detector.detect_cv2_imgs(imgs)
        img_disps = [detector.draw_bboxes(img, detections)
                     for detections in imgs_detections]
        img_disp = np.hstack(img_disps)

    # Plot
    lib_plot.cv2_imshow(img_disp, time_ms=0)

    # Save result
    OUTPUT_FOLDER = args.output_folder
    print("Result images are saved to: " + OUTPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    cv2.imwrite(OUTPUT_FOLDER +
                os.path.basename(args.image_filename), img=img_disp)


if __name__ == '__main__':
    node_name = "yolo_detection_server"
    rospy.init_node(node_name)
    args = set_inputs()
    main(args)
    rospy.logwarn("Node `{}` stops.".format(node_name))
