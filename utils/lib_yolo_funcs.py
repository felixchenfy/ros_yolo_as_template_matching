# -*- coding: future_fstrings -*-
from __future__ import division

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)

import sys
if sys.version[0:3] == "2.7":
    from src.PyTorch_YOLOv3.models import Darknet
    from src.PyTorch_YOLOv3.utils.utils import non_max_suppression, load_classes
    from src.PyTorch_YOLOv3.utils.datasets import ImgfolderDataset
else:
    from src.PyTorch_YOLOv3.models import Darknet
    from src.PyTorch_YOLOv3.utils.utils import non_max_suppression, load_classes
    from src.PyTorch_YOLOv3.utils.datasets import ImgfolderDataset
    
from utils.lib_yolo_datasets import ImgfolderDataset, UsbcamDataset, VideofileDataset
from utils.lib_common_funcs import Timer

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

# ------------------------- Library functions -------------------------



def tensor_images_to_list_numpy_images(tensor_imgs):
    imgs = tensor_imgs.permute(0, 2, 3, 1).data.numpy() # RGB, float, (20, H, W, 3)
    imgs = [img for img in imgs] # list of numpy image
    return imgs
        
        
def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    ''' This is copied from src/PyTorch_YOLOv3/utils/utils.py '''
    
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
        imgs: Batch x (3xHxW), tensor, rgb
        img_size: int
    Output:
        (let Z = img_size)
        yoloimgs: Batch x (3xZxZ), tensor, rgb
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
    if 0: # test result
        for label, pos in labels_and_pos:
            print("Detect '{}', pos = {}".format(label, pos))
    return labels_and_pos 

def create_model(args):
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(args.model_def, img_size=args.img_size).to(device)

    # Load darknet weights
    if args.weights_path.endswith(".weights"):
        model.load_darknet_weights(args.weights_path)
    else: # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))

    model.eval()  # Set in evaluation mode

    return model 

def set_dataloader(args):

    print(f"Load data from: {args.data_source}; Data path: {args.image_data_path}")
    
    if args.data_source == "folder":
        dataloader = DataLoader(
            ImgfolderDataset(args.image_data_path, img_size=args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
            # num_workers=args.n_cpu, # This causes bug in vscode (threading problem). I comment this out during debug.
        )

    elif args.data_source == "video":

        dataloader = DataLoader(
            VideofileDataset(args.image_data_path, img_size=args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
        )
        
    elif args.data_source == "webcam":
        dataloader = DataLoader(
            UsbcamDataset(max_framerate=10, img_size=args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
        )
        
    else:
        raise ValueError("Wrong data source for yolo")  
    return dataloader


def detect_targets(args, model, 
                   rgb_imgs, # Batch x (3xHxW), tensor, rgb
                   if_single_instance=False, # single instance for each class
    ):
    '''
    Output:
        detections: [bbox, conf, cls_conf, cls_pred]
            where: bbox = [x1, y1, x2, y2] represented in the original image coordinate
    '''
        
    Z = args.img_size
    yolo_imgs = rgbimgs_to_yoloimgs(rgb_imgs, Z) # [B, 3, W, H] --> [B, 3, Z, Z]
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    imgs_on_gpu = Variable(yolo_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        N_info = 7 # format of imgs_detections[jth_img]: x1, y1, x2, y2, conf, cls_conf, cls_pred
        imgs_detections = model(imgs_on_gpu)
        imgs_detections = non_max_suppression(imgs_detections, args.conf_thres, args.nms_thres)
        
        # convert to numpy array
        imgs_detections = [d.numpy() if d is not None else None for d in imgs_detections]
        
        
    # Sort detections based on confidence; 
    # Convert box to the current image coordinate
    for jth_img in range(len(imgs_detections)):
        if imgs_detections[jth_img] is None: continue
    
        # sort
        detections = sorted(imgs_detections[jth_img], key=lambda x: x[5])
        detections = np.array(detections)
        
        # change bbox pos to yoloimg
        detections = rescale_boxes(detections, args.img_size, rgb_imgs[jth_img].shape[:2])
        
        # save result
        imgs_detections[jth_img] = detections
        
    # Remove duplicated objects
    # (under the assumption that each class has only one instance for each image)
    if if_single_instance:
        for jth_img, jth_detections in enumerate(imgs_detections):
            if imgs_detections[jth_img] is None: continue
            detected_objects = set()
            jth_unique_detections = []
            for kth_object in jth_detections:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = kth_object
                if cls_pred not in detected_objects: # Add object if not detected before
                    detected_objects.add(cls_pred)
                    jth_unique_detections.append(kth_object)
            imgs_detections[jth_img] = np.array(jth_unique_detections)
    
    return imgs_detections