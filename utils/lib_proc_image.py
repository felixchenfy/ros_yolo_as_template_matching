# -*- coding: future_fstrings -*-
from __future__ import division

import cv2 
import numpy as np 



# Basics =============================================
import numpy as np
import cv2
import math


def load_image_to_binary(filename):
    ''' Load an image and convert to a binary image.
    Thresholding on alpha channel, or gray scaled image.
    '''
    ''' How to add transparency channel?
    https://sirarsalih.com/2018/04/23/how-to-make-background-transparent-in-gimp/
    https://help.surveyanyplace.com/en/support/solutions/articles/35000041561-how-to-make-a-logo-with-transparent-background-using-gimp-or-photoshop
    '''
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED) # read default format
    threshold = 127
    # print(f"image shape {img.shape}, filename: {filename}")
    
    if len(img.shape) == 2: # gray image
        mask = img > threshold
    elif img.shape[2] == 3: # color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = img > threshold
    elif img.shape[2] ==4: # image with a transparent channel 
        mask = img[:, :, -1] > threshold
    return mask


def add_mask(image, mask):
    ''' In image, set mask region to BLACK or GRAY'''
    BLACK, GRAY = 0, 127
    image = image.copy()
    idx = np.where(mask==0)
    image[idx] = GRAY
    return image 
        
def add_color_to_mask_region(img, mask, channel, mask_value = 1):
    ''' In image, for the mask region, add color to certain channel '''
    img_disp = img.copy()
    i_sub = img_disp[..., channel]
    i_sub[np.where(mask==mask_value)] = 255
    return img_disp

    
def getBbox(mask, norm=False):
    ''' Return normalized pos of the white region in mask.
    format: (x, y, w, h), same as Yolo '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        if norm:
            r, c = mask.shape[0:2]
            x = (cmin + cmax)/2/c
            y = (rmin + rmax)/2/r
            w = (cmax - cmin)/c
            h = (rmax - rmin)/r
            return x, y, w, h
        else:
            return rmin, rmax, cmin, cmax
    except:
        return None, None, None, None
        

def crop_image(img, rmin, rmax, cmin, cmax):
    if len(img.shape)==2:
        return img[rmin:rmax, cmin:cmax]
    else:
        return img[rmin:rmax, cmin:cmax, :]

def get_mask_region(img0, mask0):
    rmin, rmax, cmin, cmax = getBbox(mask0)
    img = crop_image(img0, rmin, rmax, cmin, cmax)
    mask = crop_image(mask0, rmin, rmax, cmin, cmax)
    return img, mask

def cvt_binary2color(mask):
    ''' Convert mask image to 3-channel color image by stacking 3 channels'''
    mask = mask * 255
    color = np.stack((mask, mask, mask), axis=2)
    return color 
