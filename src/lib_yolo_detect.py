# -*- coding: future_fstrings -*-
from __future__ import division

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)

import sys
from src.PyTorch_YOLOv3.models import Darknet
from src.PyTorch_YOLOv3.utils.utils import non_max_suppression, load_classes
from src.PyTorch_YOLOv3.utils.datasets import ImgfolderDataset
    
from utils.lib_yolo_datasets import ImgfolderDataset, UsbcamDataset, VideofileDataset
from utils.lib_common_funcs import Timer
from utils.lib_yolo_plot import Yolo_Detection_Plotter_CV2
import utils.lib_common_funcs as cf
from config.config import read_all_args

import os
import sys
import time
import datetime
import argparse
import cv2 
import numpy as np 
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def tensor_images_to_list_numpy_images(tensor_imgs):
    '''
    Arguments:
        tensor_imgs {tensor, BxCxHxW}
    Return:
        list_of_imgs {list of numpy images}
    '''
    imgs = tensor_imgs.permute(0, 2, 3, 1).data.numpy() # convert to: RGB, float, (20, H, W, 3)
    list_of_imgs = [img for img in imgs] # convert to: list of numpy images
    return list_of_imgs
        
        
def rescale_boxes(boxes, current_dim, original_shape):
    ''' Rescales bounding boxes to the original shape 
        This is copied from src/PyTorch_YOLOv3/utils/utils.py
    '''
    
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def resize(image, size):
    ''' Resize image to `size` '''
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def rgbimg_to_yoloimg(img, img_size):
    '''
    Input:
        img: 3xHxW, tensor, rgb
        img_size: int
    Output: 
        (let Z = img_size)
        img: 3xZxZ, tensor, rgb
    '''
    # img = np.moveaxis(img, -1, 0)  # no need for this. torchvision.transforms does this for us.
    # img = transforms.ToTensor()(img) # numpy, HxWx3 --> tensor, 3xHxW
    # img = img[np.newaxis, ...] # no need for this. DataLoader itself will add the additional channel.
        
    # Pad to square resolution
    img, _ = pad_to_square(img, 0) # 3 x H(W) x H(W)
    
    # Resize
    img = resize(img, img_size) # 3 x img_size x img_size

    return img

def rgbimgs_to_yoloimgs(imgs, img_size):
    '''
    Input:
        imgs: Batch x (3xHxW), tensor, rgb, uint8
        img_size: int
    Output:
        (let Z = img_size)
        yoloimgs: Batch x (3xZxZ), tensor, rgb, float
    '''
    imgs = imgs.type(torch.float32)
    imgs = imgs.permute(0, 3, 1, 2) # [B, W, H, 3] --> [B, 3, W, H]
    imgs /= 255.0
    yoloimgs = [rgbimg_to_yoloimg(img, img_size) for img in imgs]
    yoloimgs = torch.stack((yoloimgs))
    return yoloimgs

# ------------------ Main functions used for inference ------------------

def detetions_to_labels_and_pos(self, detections, classes):
    ''' 
    Input:
        detections: the output of "detect_targets()" 
    '''
    labels_and_pos = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        label = classes[int(cls_pred)]
        pos = (int((x1+x2)/2), int((y1+y2)/2))
        labels_and_pos.append((label, pos))
    return labels_and_pos 

def create_model(weights_path, f_yolo_config, img_size):

    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(f_yolo_config, img_size=img_size).to(device)

    # Load darknet weights
    if weights_path.endswith(".weights"):
        model.load_darknet_weights(weights_path)
    else: # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    return model 

def set_dataloader(src_data_type, image_data_path, img_size, batch_size, n_cpu):

    print(f"Load data from: {src_data_type}; Data path: {image_data_path}")
    
    if src_data_type == "folder":
        dataloader = DataLoader(
            ImgfolderDataset(image_data_path, img_size=img_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_cpu,
        )

    elif src_data_type == "video":
        dataloader = DataLoader(
            VideofileDataset(image_data_path, img_size=img_size),
            batch_size=batch_size,
            shuffle=False,
        )
        
    elif src_data_type == "webcam":
        dataloader = DataLoader(
            UsbcamDataset(max_framerate=10, img_size=img_size),
            batch_size=batch_size,
            shuffle=False,
        )
        
    else:
        raise ValueError("Wrong data source for yolo")  
    return dataloader


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def detect_targets(args_inference, model, 
                   rgb_imgs, # Batch x (3xHxW), tensor, rgb, uint8
                   is_one_obj_per_class=False, # single instance for each class
    ):
    '''
    Output:
        detections: [bbox, conf, cls_conf, cls_pred]
            where: bbox = [x1, y1, x2, y2] is represented in the original image coordinate
    '''
        
    # -- Convert images to required type
    Z = args_inference.img_size
    yolo_imgs = rgbimgs_to_yoloimgs(rgb_imgs, Z) # [B, 3, W, H] --> [B, 3, Z, Z], uint8 --> float
    imgs_on_gpu = Variable(yolo_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        imgs_detections = model(imgs_on_gpu)
        
    N_elements = 7 # format of imgs_detections[jth_img]: x1, y1, x2, y2, conf, cls_conf, cls_pred
    idx_conf = 5
    imgs_detections = non_max_suppression(imgs_detections, args_inference.conf_thres, args_inference.nms_thres)
    
    # convert to numpy array
    imgs_detections = [d.numpy() if d is not None else None for d in imgs_detections]
        
        
    # Sort detections based on confidence; 
    # Convert box to the current image coordinate;
    # Convert detections to 2d list
    for jth_img in range(len(imgs_detections)):
        
        if imgs_detections[jth_img] is None: # no detected object
            imgs_detections[jth_img] = []
            continue
    
        # sort
        detections = sorted(imgs_detections[jth_img], key=lambda x: x[idx_conf])
        detections = np.array(detections)
        
        # change bbox pos to yoloimg
        detections = rescale_boxes(detections, args_inference.img_size, rgb_imgs[jth_img].shape[:2])
        
        # save result
        imgs_detections[jth_img] = detections.tolist()
        
    # Remove duplicated objects in the single-instance mode
    if is_one_obj_per_class:
        for jth_img, jth_detections in enumerate(imgs_detections):
            if not imgs_detections[jth_img]:
                continue
            detected_objects = set()
            jth_unique_detections = []
            for kth_object in jth_detections:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = kth_object
                if cls_pred not in detected_objects: # Add object if not detected before
                    detected_objects.add(cls_pred)
                    jth_unique_detections.append(kth_object)
            imgs_detections[jth_img] = jth_unique_detections
    
    return imgs_detections


class ObjDetector(object):
    ''' Yolo detector for single image '''
    def __init__(self, config_path, weights_path):
        args = read_all_args(config_path)
        args_inference = cf.dict2class(args.yolo_inference)
        self.model = create_model(weights_path, args.f_yolo_config, args_inference.img_size)
        self.classes = load_classes(args.f_yolo_classes)  # Extracts class labels from file
        self.plotter = Yolo_Detection_Plotter_CV2(classes=self.classes, if_show=False)
        self.args, self.args_inference = args, args_inference
        
    def detect_cv2_img(self, cv2_img, is_one_obj_per_class=False):
        '''
        Argument:
            cv2_img {a clor image read from cv2.imread}
        Return:
            detections {2d list}: Each element is a 1D list indicating the detected object
                                  [[x1, y1, x2, y2, conf, cls_conf, cls_pred], [...], ...],
                                  where (x1, yi) represents in the original image coordinate
                                  
        '''
        # Change format to the required one: bgr to rgb, numpy to tensor, unsqueeze 0
        imgs = self._cv2_to_torch_img(cv2_img) 
    
        # Detect
        imgs_detections = detect_targets(
            self.args_inference, self.model, imgs, is_one_obj_per_class)
        
        # Return
        detections = imgs_detections[0] # there is only 1 image here
        return detections 
    
    def detect_cv2_imgs(self, cv2_imgs, is_one_obj_per_class=False):
        imgs = self._cv2_to_torch_imgs(cv2_imgs)
        imgs_detections = detect_targets(
            self.args_inference, self.model, imgs, is_one_obj_per_class)
        return imgs_detections

    def detect_torch_imgs(self, torch_imgs, is_one_obj_per_class=False):
        return detect_targets(
            self.args_inference, self.model, torch_imgs, is_one_obj_per_class)
        
    def draw_bboxes(self, img, detections):
        '''
        Arguments:
            detections {Nx7 arrays}: Info of each obj: Bbox(4), conf, cls_conf, cls_idx.
                This can be {2d list} or {np.ndarray} or {torch.Tensor}
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_disp = self.plotter.plot(img, detections, if_print=False)
        return img_disp
    
    def _cv2_to_torch_img(self, img): # Output: [1, W, H, 3], tensor, rgb
        ''' Change image format to what the detector requires. '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = torch.from_numpy(img).unsqueeze(0)
        return imgs 
    
    def _cv2_to_torch_imgs(self, imgs): # Output: [B, W, H, 3], tensor, rgb
        for i in range(len(imgs)):
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        torch_imgs = torch.from_numpy(np.array(imgs))
        return torch_imgs
            