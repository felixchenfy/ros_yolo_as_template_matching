''' Datasets for image augmentation '''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import glob 
import os
import cv2 

import utils.lib_common_funcs as cf
import utils.lib_proc_image as pi

def get_label(filename):
    ''' Get label from filename.
    e.g.: /folder/bottle_1.jpg --> bottle
    ''' 
    
    if '/' in filename: 
        filename = filename.split('/')[-1] # /folder/bottle_1.jpg --> bottle_1.jpg
    if '_' in filename:
        label = filename.split('_')[0]
    else:
        label = filename
    return label

class YoloLabels(object):
    def __init__(self, args):

        filename = args.f_yolo_classes

        with open(filename, 'r') as f:
            labels = [line.rstrip() for line in f]

        self.filename = filename
        self.labels = labels
    
    def parse_label(self, filename):
        
        label = get_label(filename)
        
        assert label in self.labels, "Image has wrong label: {}".format(filename)
        
        label_idx = self.labels.index(label)
        
        return label, label_idx
    
class BackgroundDataset(object):
    
    def __init__(self, args,
            resize_to_rows=None,
            preload_all_images=True,
            ):
        
        # Settings
        img_folder = args.f_background
        self.preload_all_images = preload_all_images
        self.resize_to_rows = resize_to_rows
        
        # Read image names
        self.fnames_img = cf.get_filenames(img_folder, file_types=('*.jpg', '*.png'))
        N = len(self.fnames_img)
        assert N>0, "Background images are empty. You must put them in 'data/xxxx/background/' folder."
        
        # Load images
        if preload_all_images:
            self.imgs = [self.load_ith_image(i)
                for i in range(N)]
        else:
            self.imgs = []
    
    def load_ith_image(self, i):
        I = cv2.imread(self.fnames_img[i], cv2.IMREAD_COLOR)
        rows = self.resize_to_rows
        if rows:
            r0, c0 = I.shape[:2]
            I = cv2.resize(I, (int(c0*rows/r0), rows))
        return I
    
    def __len__(self):
        return len(self.fnames_img)
    
    def __getitem__(self, i):
        if self.preload_all_images:
            return self.imgs[i]
        else:
            return load_ith_image(i)
    
class TemplatesDataset():
    def __init__(self, args,
            preload_all_images=True,
            crop_mask=True, # Crop out only a sub rectangular white region inside the mask
            ):
        
        # Settings
        img_folder = args.f_template_img
        mask_folder = args.f_template_mask 
        self.preload_all_images = preload_all_images
        self.crop_mask = crop_mask
        
        # Read image filenames
        fnames_img = cf.get_filenames(img_folder, file_types=('*.jpg', '*.png'))
        fnames_mask = cf.get_filenames(mask_folder, file_types=('*.jpg', '*.png'))
        
        # Check if files are matched
        def get_basename(fname):
            path, name, ext = cf.split_name(fname)
            return name 
        fnames_img_basename = [get_basename(fname) for fname in fnames_img]
        fnames_mask_basename = [get_basename(fname) for fname in fnames_mask]
        if fnames_img_basename != fnames_mask_basename:
            print(f"fnames_img_basename = {fnames_img_basename}")
            print(f"fnames_mask_basename = {fnames_mask_basename}")
            raise ValueError("Template images and masks have mis-matched names.")
        
        # Save vars
        self.num_templates = len(fnames_img)
        self.fnames_img = fnames_img
        self.fnames_mask = fnames_mask
        
        # Load images
        self.imgs = []
        self.masks = []
        if preload_all_images:
            for i in range(self.num_templates):
                img, mask = self.load_ith_image(i)
                self.imgs.append(img)
                self.masks.append(mask)
    
    def get_ith_filenames(self, i, base_name_only=False):
        if base_name_only:
            fimg = self.fnames_img[i].split('/')[-1]
            fmask = self.fnames_mask[i].split('/')[-1]
        else:
            fimg = self.fnames_img[i]
            fmask = self.fnames_mask[i]
        return fimg, fmask
    
    def load_ith_image(self, i):
        fimg, fmask = self.get_ith_filenames(i)
        img = cv2.imread(fimg, cv2.IMREAD_COLOR) # read as color image
        mask = pi.load_image_to_binary(fmask)
        if self.crop_mask: 
            img, mask = pi.get_mask_region(img, mask)
        return img, mask 
    
    def __len__(self):
        return self.num_templates
    
    def __getitem__(self, i):
        if self.preload_all_images:
            return self.imgs[i], self.masks[i]
        else:
            return load_ith_image(i)
            
    