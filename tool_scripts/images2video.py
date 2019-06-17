import numpy as np
import cv2
import sys, os, time
import numpy as np
import simplejson
import sys, os
import csv
import glob 
ROOT = os.path.dirname(os.path.abspath(__file__))+"/"

# Settings
image_folder =  "/home/feiyu/catkin_ws/src/simon_says/src/detection/yolo/data/digits_eval2/"
video_name =    "/home/feiyu/catkin_ws/src/simon_says/src/detection/yolo/video.avi"

fnames = sorted(glob.glob(image_folder + "*.png"))
N = len(fnames)
framerate = 10
FASTER_RATE = 1

# Read image and save to video'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
for i, name in enumerate(fnames):
    frame = cv2.imread(name)
    if i==0:
        width = frame.shape[1]
        height = frame.shape[0]
        video = cv2.VideoWriter(video_name, fourcc, framerate, (width,height))
    print(f"Processing the {i}/{len(fnames)}th image")
    if (1+i) % FASTER_RATE == 0:
        video.write(frame)

cv2.destroyAllWindows()
video.release()