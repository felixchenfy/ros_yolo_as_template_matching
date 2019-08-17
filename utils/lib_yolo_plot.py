# -*- coding: future_fstrings -*-
from __future__ import division

''' API for plotting:

-- class Yolo_Detection_Plotter_PLT:
    Use plt to plot result. Very slow. 0.35s per image.

-- class Yolo_Detection_Plotter_CV2:
    Use cv2 to plot result. Fast. 0.02s
    
'''

import numpy as np 
import time
import cv2 
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

''' Notes:
cv2.rectangle Parameters:
    img: Image.
    pt1: left-up        coordinate (x, y) of type int
    pt2: right-bottom   coordinate (x, y) of type int
        or using "rec" to replace "pt1" and "pt2"
    lineType: Type of the line:
        8 (or omitted) - 8-connected line.
        4 - 4-connected line.
        CV_AA - antialiased line.
    shift: Number of fractional bits in the point coordinates.


cv2.putText Parameters:
    img:	    Image.
    text:	    Text string to be drawn.
    org:	    Bottom-left corner of the text string in the image.
    fontFace:	Font type, see HersheyFonts.
    fontScale:	Font scale factor that is multiplied by the font-specific base size.
    color:	    Text color.
    thickness:	Thickness of the lines used to draw a text.
    lineType:	Line type. See LineTypes
    bottomLeftOrigin:	When true, the image data origin is at the bottom-left corner. 
        Otherwise, it is at the top-left corner. Default false.
        If true, the words will be upside-down.
'''
        

def put_text_with_background_color(
    img, text, text_pos, text_color, bg_color, 
    font_scale=2, thickness=2, font=cv2.FONT_HERSHEY_PLAIN
    ):
    ''' cv2.put_text to draw text, and cv2.rectangle to draw background color '''
    ''' Modified from: https://gist.github.com/aplz/fd34707deffb208f367808aade7e5d5c '''
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # make the coords of the box with a small padding of two pixels
    x, y = text_pos
    box_coords = ((x, y), (x + text_width - 2, y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), fontFace=font, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)


# Draw onto image the bounding boxes and texts of the detection results (Very Very Slow!)
class Yolo_Detection_Plotter_PLT(object):
    def __init__(self, if_show):
        self.cmap = plt.get_cmap("tab20b")
        self.colors = [self.cmap(i) for i in np.linspace(0, 1, 20)]
        self.if_show = if_show 
        
        # Init figure
        if if_show: plt.ion()
        fig = plt.figure(figsize=(12, 8))
        self.ax = fig.add_subplot(111) # or: ax = plt.gca()
    
    def savefig(self, filename):
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)
        
    def plot(self, img, detections, classes):
        plt.cla()
        img = img/255.0 # rgb, uint8 --> rgb, float
        self._plot(img, detections, classes)
        if self.if_show: plt.pause(0.001)
    
    def close(self):
        plt.close()
        
    def _plot(self, img, detections, classes):
        colors = self.colors
        
        # Plot image
        ax = self.ax 
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if (detections is not None) and (len(detections) > 0):
            # unique_labels = detections[:, -1].cpu().unique() # tensor version
            unique_labels = np.unique(detections[:, -1]) # numpy version
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="bottom",
                    bbox={"color": color, "pad": 0},
                    fontsize=20,
                )
                print("box: {}".format(bbox))

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Draw onto image the bounding boxes and texts of the detection results 
class Yolo_Detection_Plotter_CV2(object):
    def __init__(self, classes, if_show=True, 
                 cv2_waitKey_time=1, # (ms) 
                 window_name="Yolo detection results",
                 resize_scale=1, # resize image larger
                 ):
        self.cmap = plt.get_cmap("tab20")
        self.colors = [self.cmap(i) for i in np.linspace(0, 1, 20)]
        self.colors = (np.array(self.colors)*255).astype(np.uint8) # to uint8
        self.window_name = window_name
        self.cv2_waitKey_time = cv2_waitKey_time
        self.resize_scale = resize_scale
        self.if_show = if_show 
        self.img_disp = None # The image wait to be plot: float, 0~1, RGB. 
        self.classes = classes

    def savefig(self, filename):
        cv2.imwrite(filename, self.img_disp)

    def plot(self, img, detections, img_channels='rgb', if_plot_center=True, if_print=True):
        '''
        Input:
            img: 3 channels, rgb, uint8
            detections: bbox pos in image coordinate
        Output:
            self.img_disp: 3 channels, bgr, uint8
        '''
        if img_channels in ["rgb", "RGB"]: 
            self.img_disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # change color to BGR. This is what cv2.imshow needs.
        else:
            self.img_disp = img.copy()
            
        self._plot_onto_image(self.img_disp, detections, if_plot_center, if_print)
        
        if self.resize_scale != 1:
            self.img_disp = cv2.resize( # make the drawing larger
                self.img_disp, dsize=(0, 0), fx=self.resize_scale, fy=self.resize_scale) 
        
        if self.if_show: 
            cv2.imshow(self.window_name, self.img_disp)
            cv2.waitKey(self.cv2_waitKey_time)
        return self.img_disp
        
    def close(self):
        cv2.destroyAllWindows()
        
    def _plot_onto_image(self, img, detections, if_plot_center, if_print):
        colors = self.colors
        classes = self.classes
        if isinstance(detections, list):
            detections = np.array(detections)
            
        # Draw bounding boxes and labels of detections
        if (detections is not None) and (len(detections) > 0):
            # unique_labels = detections[:, -1].cpu().unique() # tensor version
            unique_labels = np.unique(detections[:, -1]) # numpy version
            n_cls_preds = len(unique_labels)
            
            # bbox_colors = random.sample(colors, n_cls_preds) # this makes color changes. not good.
            bbox_colors = colors
            
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections: 
                if if_print:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], conf.item()))
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])].tolist()
                xc, yc = (x1+x2)/2, (y1+y2)/2
                
                # Draw rectangular
                x1, y1, width, height = map(int, [x1, y1, x2-x1, y2-y1]) # to int
                bbox = (x1, y1, width, height)
                img = cv2.rectangle(img, rec=bbox, color=color, thickness=2)
                
                # Draw a circle at center
                if if_plot_center:
                    p_cen = (int((x1+x2)/2), int((y1+y2)/2))
                    cv2.circle(img, p_cen, radius=1, color=color, 
                            thickness=2, lineType=cv2.LINE_AA)
                    cv2.circle(img, p_cen, radius=3, color=[0,0,255], 
                            thickness=1, lineType=cv2.LINE_AA)
                # Draw text
                put_text_with_background_color(
                    img, 
                    text=classes[int(cls_pred)], 
                    text_pos=(x1, y1),
                    text_color=(255, 255, 255), # white 
                    bg_color=color,
                    font_scale=1.7, 
                    thickness=1, 
                    font=cv2.FONT_HERSHEY_PLAIN)
                if if_print: 
                    print("\t  box: center=({:.1f}, {:.1f}), w={:.1f}, h={:.1f}".format(xc, yc, width, height))
            # end of "if detections is not None:"
        

