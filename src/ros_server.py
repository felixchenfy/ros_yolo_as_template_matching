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
import os
import sys
import argparse
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
                        required=True,
                        help="ROS topic: src image for object detection.")
    parser.add_argument("-o", "--dst_topic_img", type=str,
                        required=True,
                        help="ROS topic: dst image that shows the results.")
    parser.add_argument("-t", "--dst_topic_res", type=str,
                        required=True,
                        help="ROS topic: dst results of each object's label, confidence and bbox.")
    args = parser.parse_args()
    return args


class DetectionResultsPublisher(object):
    def __init__(self, topic_name, classes, queue_size=10):
        self._pub = rospy.Publisher(
            topic_name, DetectionResults, queue_size=queue_size)
        self._classes = classes

    def publish(self, detections):
        '''
        Arguments:
            detections {Nx7 arrays}: Info of each obj: Bbox(4), conf, cls_conf, cls_idx.
                This can be {2d list} or {np.ndarray} or {torch.Tensor}
        '''
        res = DetectionResults()
        res.N = len(detections)
        for x1, y1, x2, y2, conf, cls_conf, cls_idx in detections:
            label = self._classes[int(cls_idx)]
            confidence = cls_conf
            bbox = [x1, y1, x2, y2]

            res.labels.append(label)
            res.confidences.append(confidence)
            res.bboxs.extend(bbox)
        self._pub.publish(res)
        return


def main(args):

    # -- Init detector
    rospy.loginfo("Initializing YOLO detector ...")
    detector = lib_yolo_detect.ObjDetector(args.config_path, args.weights_path)
    classes = detector.classes
    rospy.loginfo("Initializing completes.")

    # -- Input ROS topic.
    sub_img = ColorImageSubscriber(args.src_topic_img)
    rospy.loginfo("Subscriber image from: " + args.src_topic_img)

    # -- Output ROS topics.
    pub_img = ColorImagePublisher(args.dst_topic_img)
    rospy.loginfo("Publish result image to: " + args.src_topic_img)
    pub_res = DetectionResultsPublisher(args.dst_topic_res, classes)
    rospy.loginfo("Publish detection results to: " + args.dst_topic_res)

    # -- Subscribe images and detect.
    rospy.loginfo("Start waiting for image and doing detection!")
    while not rospy.is_shutdown():
        if sub_img.has_image():

            # -- Read image from the subscription queue.
            img = sub_img.get_image()
            timer = Timer()

            # -- Detect objects.
            rospy.loginfo("=================================================")
            rospy.loginfo("Received an image. Start object detection.")

            detections = detector.detect_cv2_img(img)
            img_disp = detector.draw_bboxes(img, detections)

            # -- Print results.
            for x1, y1, x2, y2, conf, cls_conf, cls_idx in detections:
                label = classes[int(cls_idx)]
                print("  Label = {}; Conf = {}; Bbox = {}".format(
                    label, cls_conf, (x1, y1, x2, y2)))
            timer.report_time(msg="Detection")

            # -- Publish result.
            pub_img.publish(img_disp)
            pub_res.publish(detections)
            rospy.loginfo("Publish results completes.")
            rospy.loginfo("-------------------------------------------------")
            rospy.loginfo("")


if __name__ == '__main__':
    node_name = "yolo_detection_server"
    rospy.init_node(node_name)
    rospy.loginfo("ROS node starts: " + node_name)
    args = set_inputs()
    main(args)
    rospy.logwarn("Node `{}` stops.".format(node_name))
