# -*- coding: future_fstrings -*-
from __future__ import division
import numpy as np
import cv2

from skimage.transform import warp
from skimage.transform import AffineTransform
import types

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
        
def dict2class(args_dict):
    args = SimpleNamespace()
    args.__dict__.update(**args_dict)
    return args 


def increase_color(img, brightness_offset):
    img = img + brightness_offset # data type is now np.float
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img 

class ImgAugmenter(object):
    def __init__(self, args_dict):
        self._init_args()
        self._set_args_for_background_augment() # This is fixed. Please modify directly in this function
        self.update_args_for_template_augment(args_dict)
        

    def _init_args(self):
        args_dict = {
            "rela_size" : (0.1, 0.3), # desired object size relative to the background image
            "rescale" : (1, 1), # scale again randomly
            "rotation" : (0, 0),
            "shear" : (0, 0),
            "translate_x" : (0, 1),
            "translate_y" : (0, 1),
            "brightness_offset" : (0, 0),
        }
        self.args = dict2class(args_dict)
        
    def _set_args_for_background_augment(self):
        
        # Add effects to the background image 
        
        self.augseq_transforms = iaa.Sequential([ 
            iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="edge"),
            iaa.Affine(
                rotate=(-10, 10), 
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                shear=(-1, 1),
            ),
        ])
        
        self.augseq_noises = iaa.Sequential([ 
            iaa.Multiply((0.5, 1.5), per_channel=False),
            iaa.Add((-10, 10), per_channel=True),
            iaa.AdditiveGaussianNoise(scale=0.03*255),
            iaa.GaussianBlur(sigma=(0, 0.2))
            #iaa.Sharpen(alpha=0.5)
            #iaa.CoarseDropout(0.1, size_percent=0.2),
            #iaa.ElasticTransformation(alpha=10, sigma=1)
        ])
        
    def update_args_for_template_augment(self, args_dict):
        for key, val in args_dict.items():
            if hasattr(self.args, key):
                setattr(self.args, key, val)

    def _rand_values_from_args(self, args):
        ''' generate a random value given a range (l, r) from each argument in args '''

        # Generate random value
        args_dict = {}
        for arg_name, arg_val in args.__dict__.items():
            l, r = arg_val 
            if l == r:
                args_dict[arg_name] = l 
            else:
                args_dict[arg_name] = np.random.uniform(l, r)
        
        # Update some args
        args_dict["rotation"] *= np.pi/180
        
        # Output namedtuple
        args = dict2class(args_dict)
        return args

    def augment_by_transform(self, img_bg):
        img_bg = self.augseq_transforms.augment_image(img_bg)
        return img_bg 
    
    def augment_by_noises(self, img_bg):
        img_bg = self.augseq_noises.augment_image(img_bg)
        return img_bg 
    
    def put_object_onto_background(self,
            img_obj, # image of the object
            mask_obj, # mask of the object
            img_bg, # bgd -> background
        ):
        ''' Put object onto background image with random transformations specified by args '''
        
        img_obj = img_obj.copy()
        mask_obj = mask_obj.copy()
        img_bg = img_bg.copy()
        
        new_bg, new_mask = self._put_object_onto_background(img_obj, mask_obj, img_bg)
        
        # Augment again to the big image
        # (Not use this: This causes problem when adding several objects.)
        if 0: 
            segmap = SegmentationMapOnImage(new_mask, nb_classes=2, shape=new_bg.shape)
            new_bg, new_mask = self.seq(image=new_bg, segmentation_maps=segmap)
            new_mask = new_mask.get_arr_int().astype(np.uint8)
        
        return new_bg, new_mask

    def _put_object_onto_background(self, img_obj, mask_obj, img_bg):
        ''' Put object onto background image with random transformations specified by args '''
        
        # Generate random argument values
        args = self._rand_values_from_args(self.args)

        # Change image color
        img_obj = increase_color(img_obj, args.brightness_offset)

        # Set scalling
        def get_len(img): # mean length of an image 
            return np.mean(img.shape[0:2])
        scale = args.rescale * (get_len(img_bg) * args.rela_size) / get_len(img_obj)
        
        # Set translation for the scaled object
        obj_row, obj_col = img_obj.shape[0]*scale, img_obj.shape[1]*scale, 
        obj_len = (obj_row + obj_col) / 2
        b = (obj_len / 2) / get_len(img_bg)
        b *= 1.1 # scale some factor, to avoid the object going out of image
        x0, y0, w0, h0 = b, b, 1-2*b, 1-2*b # rectangular in img_bg to put the object 
        tx = (x0 + w0 * args.translate_x) * img_bg.shape[1]
        ty = (y0 + h0 * args.translate_y) * img_bg.shape[0]

        # Set transformation matrix
        t1 = AffineTransform(
            translation=(-obj_col/2, -obj_row/2)) # move img_obj to origin
        t2 = AffineTransform(
            scale=(scale, scale), rotation=args.rotation, shear=args.shear)
        t3 = AffineTransform(
            translation=(tx, ty))
        def combine(*args):
            t0 = args[0].params
            for ti in args[1:]:
                t0 = t0.dot(ti.params)
            return AffineTransform(t0)
        tform = combine(t3, t2, t1)
        
        # Print transform
        if 0: 
            print("transf: rescale={}, scale_y={}, rot={}, shear={}, tx={}, ty={}".format(
                args.rescale, args.rotation, args.shear,tx,ty))
            print(tform.params)

        # Transform
        bgd_shape = img_bg.shape[0:2]
        new_obj = warp(img_obj, tform.inverse, output_shape=bgd_shape)
        new_mask = warp(mask_obj, tform.inverse, output_shape=bgd_shape)

        # Convert data format back to uint8
        new_obj = np.round(new_obj*255).astype(np.uint8) # format: BRG, same as input image
        new_mask = np.round(new_mask).astype(np.uint8)

        # Get new background image
        new_bg = img_bg
        idx = np.where(new_mask==1)
        new_bg[idx] = (new_obj[idx]).astype(np.uint8)

        return new_bg, new_mask
