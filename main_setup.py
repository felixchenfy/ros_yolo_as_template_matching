#!/usr/bin/env python3
''' 
Setup all things. See help:
$ python src/setup.py -h
'''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/" # root of the project
    sys.path.append(ROOT)
    
import glob
import numpy as np 
import cv2
import datetime
import time 
import shutil

if 1: # import configurations
    import argparse
    from config.config import read_all_args
    from config.yolo_config import YoloConfig
    
if 1: # import my libs
    import utils.lib_common_funcs as cf
    import utils.lib_proc_image as pi 
    from utils.lib_datasets import TemplatesDataset, BackgroundDataset, YoloLabels, get_label
    from utils.lib_plot import show, draw_bbox
    from utils.lib_augment_image import ImgAugmenter
    
def get_time():
    s=str(datetime.datetime.now())[5:].replace(' ','-').replace(":",'-').replace('.','-')[:-3]
    return s # day, hour, seconds: 02-26-15-51-12-556

def write_object_labels_to_file(args):
    ''' 
    From "data/$data_name/template_img/", read in filenames to know the labels. 
    A filename should be like in this form "labelname_index.jpy", e.g.: "bottle_1.jpg"
    '''
    
    # Get object labels
    if 0: # from "data/$data_name/template_img/"
        fnames = cf.get_filenames(args.f_template_img, file_types=('*.jpg', '*.png'))
        labels = {get_label(fn) for fn in fnames} # /folder/bottle_1.jpg --> bottle
        labels = sorted(list(labels))
    else: # from configurations
        labels = args.data_labels
    print(f"Object labels: {labels}")
        
    # Write labels to txt for yolo training
    cf.write_list(filename=args.f_yolo_classes, arr=labels)
    print(f"Write labels to {args.f_yolo_classes}")
    
    return labels

def create_masked_template_to_verify(args):
    ''' 
    Read in template mask drawn by the user, verify it, and then write it to dst folder. 
    Output format: gray image with white/black values    
    '''
    
    # Settings
    folder_dst = args.f_data_dst + "masked_template/"
    
    # Check
    cf.create_folder(folder_dst)
    
    # Vars
    dataset_tp = TemplatesDataset(args) # template dataset
    
    # Start
    for i in range(len(dataset_tp)):
        
        # Read in template image and corresponding mask
        img, mask = dataset_tp.load_ith_image(i)
        
        # Create masked image
        img = pi.add_mask(img, mask) # put mask onto the image
        res_img = np.hstack((img, pi.cvt_binary2color(mask)))
        
        # Write to file
        fname = dataset_tp.get_ith_filenames(i, base_name_only=True)[0]
        cv2.imwrite(folder_dst + fname, res_img)

def augment_images(args):
    
    # Settings
    pass 

    # Check
    cf.create_folder(args.f_yolo_images)
    cf.create_folder(args.f_yolo_labels)
    cf.create_folder(args.f_yolo_images_with_bbox)
    
    # Vars
    dataset_tp = TemplatesDataset(args) # template dataset
    dataset_bg = BackgroundDataset(args, resize_to_rows=600) # background dataset
    yolo_labels = YoloLabels(args) # label parser
    aug = ImgAugmenter(args.template_aug_effects)
    
    # Functions
    def get_random_template():
        i = np.random.randint(len(dataset_tp))
        image, mask = dataset_tp[i]
        filename = dataset_tp.get_ith_filenames(i)[0]
        label, label_idx = yolo_labels.parse_label(filename)
        return image, mask, label, label_idx
    
    def get_random_background():
        i = np.random.randint(len(dataset_bg))
        return dataset_bg[i] # return image

    def random_object_number():
        l, r = args.img_aug_nums["objects_per_image"]
        return np.random.randint(l, r+1)
    
    # Start
    for ith_background in range(args.img_aug_nums["num_new_images"]):
        print("Generating {}th augmented image ...".format(ith_background))
        
        # Read background image
        bg_img = get_random_background()
        
        # Vars to store
        masks = []
        labels = []
        
        # 1st augment (apply small affine to background image)
        bg_img = aug.augment_by_transform(bg_img)

        # Add many templates onto background image
        for ith_object in range(random_object_number()):
            
            # Read template
            tp_img, tp_mask, label, label_idx = get_random_template()
            
            # put template onto the background image        
            new_bg, new_mask = aug.put_object_onto_background(tp_img, tp_mask, bg_img)
            
            # Store vars
            bg_img = new_bg
            masks.append(new_mask)
            labels.append([label_idx])
            
        # Last augment (add noise to the new background image)
        bg_img = aug.augment_by_noises(bg_img)
        
        # Get/Save/Plot bounding boxt of background image
        bg_img_with_bbox = bg_img.copy()
        for i, mask in enumerate(masks):
            
            # Get bbox
            x, y, w, h = pi.getBbox(mask, norm=True)
            
            # Store and draw
            labels[i].extend([x, y, w, h])
            draw_bbox(bg_img_with_bbox, bbox=(x, y, w, h))
            
        # Display the new background image
        if 0:
            # show((tp_img, tp_mask), figsize=(10, 5))
            show((new_mask, bg_img, bg_img_with_bbox), figsize=(15, 6), layout=(1, 3))
        
        # Write the image and its labels
        filename = "{:06d}".format(ith_background) + get_time()
        
        f_image = args.f_yolo_images + filename + ".jpg"
        f_labels = args.f_yolo_labels + filename + ".txt"
        f_image_with_bbox = args.f_yolo_images_with_bbox + filename + ".jpg"
        
        cv2.imwrite(f_image, bg_img) # new image
        cf.write_listlist(f_labels, labels) # its labels file
        cv2.imwrite(f_image_with_bbox, bg_img_with_bbox) # new image with annotated bbox on it 
        
        continue
    
    # End
    print("\n" + "-"*80)
    print("Image augmentation completes.")
    print("Generated {} images. See folder = {}".format(
        ith_background+1, args.f_data_dst))
    
    return 

def setup_train_test_txt(args):
    
    # Train/valid split and write to train.txt and valid.txt
    fnames = cf.get_filenames(args.f_yolo_images)
    rt = args.yolo["ratio_train"]
    fname_trains, fname_valids = cf.train_valid_split(fnames, ratio_train=rt)
    cf.write_list(args.f_yolo_train, fname_trains)
    cf.write_list(args.f_yolo_valid, fname_valids)
    print("Split all {} images into [train/valid={:.2f}/{:.2f}]:".format(len(fnames), rt, 1-rt))
    print("\t{}".format(args.f_yolo_train))
    print("\t{}".format(args.f_yolo_valid))
    
    # Copy images in valid.txt to the "valid_images/" folder
    if os.path.isdir(args.f_yolo_valid_images):
        shutil.rmtree(args.f_yolo_valid_images)
    cf.copy_files(fname_valids, args.f_yolo_valid_images)
    print("\tSave valid images into {}".format(args.f_yolo_valid_images))
    
def setup_yolo_files(args):
    # Write yolo.cfg, which is yolo's network configurations file
    n_labels = len(args.labels)
    yolo_config = YoloConfig(NUM_CLASSES=n_labels)
    yolo_config.write_to_file(args.f_yolo_config)
    
    # Write yolo.data, which specifies the data path to train yolo
    s = [""]*4
    s[0] = f"classes= {n_labels}"
    s[1] = f"train= {args.f_yolo_train}"
    s[2] = f"valid= {args.f_yolo_valid}"
    s[3] = f"names= {args.f_yolo_classes}"
    cf.write_list(filename=args.f_yolo_data, arr=s)
    print("Write data config to: ",args.f_yolo_data)
    
def MyBool(v):
    ''' A bool class for argparser '''
    # TODO: Add a reference
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/config.yaml", 
                        help="path to config file")
    parser.add_argument("--verify_mask", type=MyBool, default=False, 
                        help="Whether write masked templates to file for user to verify if mask is correct or not")
    parser.add_argument("--augment_imgs", type=MyBool, default=False, 
                        help="Whether do image augment and create many new images")
    parser.add_argument("--setup_train_test_txt", type=MyBool, default=False, 
                        help="Setup train.txt and valid.txt for yolo. Copy validation images to a new folder.")
    parser.add_argument("--setup_yolo", type=MyBool, default=False, 
                        help="Setup yolo.cfg, yolo.data, ")
    parser.add_argument("--create_bash_for_yolo", type=MyBool, default=False, help="Create two bash scripts for trainning yolo and doing inference: s2_train.sh & s3_inference.sh")
    args_from_command_line = parser.parse_args()
    
    # Args from configuration file
    args_from_file = read_all_args(args_from_command_line.config_file)
    
    # Combine the two
    args = args_from_command_line
    args.__dict__.update(args_from_file.__dict__)
    return args 
    
    
def main(args):
    
    
    ALL_ON = False # Turn on all functions. For debug only.
    
    if True: # Write object labels, for yolo training. This is necessary, and fast.
        labels = write_object_labels_to_file(args) 
        args.labels = labels
        
    if ALL_ON or args.verify_mask: # Write masked template, for user to verify if mask is correct
        create_masked_template_to_verify(args) 
    
    if ALL_ON or args.augment_imgs: # This takes time, so I use args to specify whether do this or not
        augment_images(args)
    
    
    if ALL_ON or args.setup_train_test_txt:
        setup_train_test_txt(args)
    
    if ALL_ON or args.setup_yolo:
        setup_yolo_files(args)
    
    if ALL_ON or args.create_bash_for_yolo:
        pass # TODO
        
if __name__ == "__main__":
    args = parse_args()
    main(args)