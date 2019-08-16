# -*- coding: future_fstrings -*-
from __future__ import division

import numpy as np 
import glob
import os 
import io, yaml 
import types 
import time
from shutil import copyfile
from datetime import datetime
import json
import yaml

# IO ----------------------------------------------------------------

def create_folder(folder):
    if folder and not os.path.exists(folder):
        print("Creating folder:", folder)
        os.makedirs(folder)
        
def get_filenames(folder, file_types=('*.jpg', '*.png')):
    filenames = []
    for file_type in file_types:
        filenames.extend(glob.glob(folder + "/" + file_type))
    filenames.sort()
    return filenames

def load_yaml(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

def write_dict(filename, dic):
    create_folder(os.path.dirname(filename))
    with open(filename, 'w') as f:
        yaml.dump(dic, f)
        # f.write(yaml.dump(dic))
        # f.write(json.dumps(dic))

def write_list(filename, arr):
    ''' Write list[] to file. Each element takes one row. '''
    create_folder(os.path.dirname(filename))
    with open(filename, mode='w') as f:
        for s in arr:
            s = s if isinstance(s, str) else str(s) # to string
            f.write(s + "\n") 

def write_listlist(filename, arrarr): 
    ''' Write list[list[]] to file. Each inner list[] takes one row. 
    This is for write yolo labels of each image
    '''
    create_folder(os.path.dirname(filename))
    with open(filename, 'w') as f:
        for j, line in enumerate(arrarr):
            line = [v if isinstance(v, str) else str(v) for v in line]
            for i, val in enumerate(line):
                if i>0:
                    f.write(" ")
                f.write(val)
            f.write("\n")

def copy_files(src_filenames, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for name_with_path in src_filenames:
        basename = os.path.basename(name_with_path)
        copyfile(src=name_with_path, dst=dst_folder + "/" + basename)

# String/List/Math ----------------------------------------------------------------

def split_name(name):
    # "/usr/lib/image.jpg" --> ["/usr/lib", "image", ".jpg"]
    pre, ext = os.path.splitext(name)
    if "/" in pre:
        p = pre.rindex('/')
        path = pre[:p] 
        name = pre[p+1:]
    else:
        path = "./"
        name = pre 
    return path, name, ext

def get_readable_time(no_blank=False):
    ''' Get the readable time of UTC 0
    '''
    ts = time.time()
    # if you encounter a "year is out of range" error the timestamp
    # may be in milliseconds, try `ts /= 1000` in that case
    if no_blank:
        readable_time = datetime.utcfromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    else:
        readable_time = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return readable_time

# Train DL ----------------------------------------------------------------------------------

def train_valid_split(filenames, ratio_train=0.8):
    N = len(filenames)
    n_train = int(N * ratio_train)
    idxs = np.random.permutation(N)
    fname_trains = [filenames[i] for i in idxs[:n_train]]
    fname_valids = [filenames[i] for i in idxs[n_train:]]
    return fname_trains, fname_valids

# ----------------------------------------------------------------------------------
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
    # args = types.SimpleNamespace() # for python 2, this doesn't work
    args = SimpleNamespace() 
    args.__dict__.update(**args_dict)
    return args 

class Timer(object):
    def __init__(self):
        self.t0 = time.time()

    def report_time(self, action):
        t = time.time() - self.t0
        t = "{:.2f}".format(t)
        print(f"'{action}' takes {t} seconds")

    def reset(self):
        self.t0 = time.time()
        
if __name__ == "__main__":
    
    def test_split_name():
        name = "/usr/lib/image.jpg"
        res = split_name(name)
        print(res)
        
    t0 = get_readable_time()
    print(t0)
    
    dic = {"da": 123, "ad": 321}
    write_dict("test.yaml", dic)