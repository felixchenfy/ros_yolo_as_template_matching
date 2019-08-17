# -*- coding: future_fstrings -*-
from __future__ import division

import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt 

def convert(img):
    '''change image color from "BGR" to "RGB" for plt.plot()'''
    if isinstance(img.flat[0], np.float):
        img = (img*255).astype(np.uint8)
    if len(img.shape)==3 and img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show(imgs, figsize=(6, 10), layout=None):
    plt.figure(figsize=figsize)
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    imgs = [convert(img) for img in imgs]
    N = len(imgs)
    
    # Set subplot size
    if layout is not None:
        r, c = layout[0], layout[1]
    else:
        if N <= 4:
            r, c = 1, N
        else:
            r, c = N//4+1, 4

    # Plot
    for i in range(N):
        plt.subplot(r, c, i+1)
        plt.imshow(imgs[i])
    plt.show()

def cv2_imshow(img, time_ms=0):
    cv2.imshow("image", img)
    cv2.waitKey(time_ms)
    cv2.destroyAllWindows()

def draw_bbox(img, bbox, color=(0,255,0), thickness=2):
    # img = img.copy()
    r, c = img.shape[:2]
    x, y, w, h = bbox  
    x0 = (x - w / 2) * c
    x1 = (x + w / 2) * c
    y0 = (y - h / 2) * r
    y1 = (y + h / 2) * r 
    x0, x1, y0, y1 = map(int, [x0, x1, y0, y1])
    img = cv2.rectangle(
        img,
        (x0, y0),
        (x1,y1),
        color=color,
        thickness=thickness)
    # return img
