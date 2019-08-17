''' Usage:
In the main script, import "read_all_args" function
In this script, modify parameters only "set_fixed_arguments" function
'''

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Set fixed arguments that doesn't need to change
def set_fixed_arguments(args):
    
    args.f_data_src = ROOT + "data/" + args.data_name + "/" # data/custom1/
    args.f_data_dst = ROOT + "data/" + args.data_name + "_generated/" # data/custom1_generated/
    args.f_data_eval = ROOT + "data/" + args.data_name + "_eval/" # data/custom1_generated/
    
    args.f_template_img = args.f_data_src + "template_img/" # data/custom1/template_img
    args.f_template_mask = args.f_data_src + "template_mask/" # data/custom1/template_mask
    args.f_background = args.f_data_src + "background/" # data/custom1/template_mask
    
    args.f_yolo_images = args.f_data_dst + "images/"
    args.f_yolo_labels = args.f_data_dst + "labels/"
    args.f_yolo_images_with_bbox = args.f_data_dst + "images_with_bbox/"
    
    args.f_yolo_classes = args.f_data_dst + "classes.names"
    args.f_yolo_train = args.f_data_dst + "train.txt"
    args.f_yolo_valid = args.f_data_dst + "valid.txt"
    args.f_yolo_valid_images = args.f_data_dst + "valid_images/"
    args.f_yolo_config = args.f_data_dst + "yolo.cfg"
    args.f_yolo_data = args.f_data_dst + "yolo.data"
# --------------------------------------------------------------------
# --------------------------------------------------------------------


if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
import utils.lib_common_funcs as cf

def read_all_args(config_file="config/config.yaml"):
    
    # Read args from yaml file
    args_dict = cf.load_yaml(config_file)
    args = cf.dict2class(args_dict)
    
    # Some fixed arguments that doesn't need change
    set_fixed_arguments(args)
    
    return args 
