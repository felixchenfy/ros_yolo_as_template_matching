# -*- coding: future_fstrings -*-
from __future__ import division

''' Datasets for image yolo '''
'''
Part of this script if copied from "src/PyTorch_YOLOv3/utils/datasets.py" and then modified
'''

import glob 
import cv2 
import numpy as np 
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import cv2 
import time


class ImgfolderDataset(Dataset):
    def __init__(self, folder_path, img_size=416, suffixes=("jpg", "png")):
        files = []
        for suffix in suffixes:
            files.extend( glob.glob(f"{folder_path}/*.{suffix}"))
        self.files = sorted(files)
        self.img_size = img_size
        print(f"The folder has {len(self.files)} images")

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        
        # Extract image as PyTorch tensor
        if 1:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H x W x 3, RGB
        else: # Alternatively, we can use: 
            img = np.array(Image.open(img_path)) # H x W x 3, RGB 
            # It returns a opened file of the img_path, 
            # and then use np.array to load in the data.
            
        return img_path, img

    def __len__(self):
        return len(self.files)


class UsbcamDataset(object):
    ''' 
    Init "torch.utils.data.DataLoader" with an instance of this class,
    and then use enumerate() to get images.
    A complete test case is in "def test_usbcam"
    '''
    
    def __init__(self, img_size=416, max_framerate=10):
        self.cam = cv2.VideoCapture(0)
        self.img_size = img_size
        self.frame_period = 1.0/max_framerate*0.999
        self.prev_image_time = time.time() - self.frame_period
        self.cnt_img = 0
        
    def __len__(self):
        return 999999
    
    def __getitem__(self, index):

        # read next image
        self.wait_for_framerate()
        ret_val, img = self.cam.read()
        self.prev_image_time = time.time()
        self.cnt_img += 1
        img_path = "tmp/{:06d}.jpg".format(self.cnt_img)
        
        # change format for yolo
        # img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H x W x 3, RGB
        return img_path, img
    
    def wait_for_framerate(self):
        t_curr = time.time()
        t_wait = self.frame_period - (t_curr - self.prev_image_time)
        if t_wait > 0:
            time.sleep(t_wait)

class VideofileDataset(object):
    ''' 
    Init "torch.utils.data.DataLoader" with an instance of this class,
    and then use enumerate() to get images.
    '''  
    def __init__(self, filename, img_size=416):
        self.cap = cv2.VideoCapture(filename)
        self.img_size = img_size
        self.cnt_img = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"The video has {self.total_frames} images")
        
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, index):
        ret_val, img = self.cap.read()
        self.cnt_img += 1
        img_path = "tmp/{:06d}.jpg".format(self.cnt_img)
        
        # change format for yolo
        # img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H x W x 3, RGB
        return img_path, img
    
def test_usbcam():
    
    def tensor_images_to_list_numpy_images(input_tensor_imgs):
        imgs = input_tensor_imgs.permute(0, 2, 3, 1).data.numpy() # RGB, float, (20, H, W, 3)
        imgs = [img for img in imgs] # list of numpy image
        return imgs


    def cv2_plot(img, wait):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("", img) # imshow, needs BRG
        cv2.waitKey(wait)
    
    
    dataloader = DataLoader(
        UsbcamDataset(),
        batch_size=1,
        shuffle=False)
    time0 = time.time()
    
    for batch_i, (imgs_path, input_tensor_imgs) in enumerate(dataloader):
        print(time.time() - time0)
        imgs = tensor_images_to_list_numpy_images(input_tensor_imgs)
        for img in imgs:
            cv2_plot(img, wait=1)
    
    cv2.destroyAllWindows()
            
if __name__=="__main__":
    from torch.utils.data import DataLoader
    test_usbcam()