# Yolo as Template Matching

# 1. Introduction
**Abstract:**  

Run 3 simple scripts to (1) augment data (by putting few template images onto backgrounds), (2) train YOLOv3, and (3) achieve object detection.

It's named `Template Matching` because only a few template images are used for training.

**Support**:
- **Language**: Python 2 or 3; ROS or not ROS. 
- **Data source** for inference: (1) An image; (2) A folder of images; (3) A video file. (4) Webcam. (5) ROS topic. 

**Demo of Data augmentation**:
![](doc/demo_of_image_aug.gif)



**Demo of Object detection:**  
<p align = "center">
  <img src = "doc/demo_1.gif" height = "240px">
  <img src = "doc/demo_2.gif" height = "240px">
</p>

The training data of the above data are: 
(1) 4 template images.   
(2) 25 background images(homes) downloaded directly from google. 
By putting templates onto backgrounds, `3000` synthsized images were created, and then traiend for `500` epochs. The workflow is shown below.

**System workflow:**  
![](doc/fig1_intro.png)



**Reference**: The core code of Yolo is copied from [this repo](https://github.com/eriklindernoren/PyTorch-YOLOv3), and is stored in [src/PyTorch_YOLOv3](src/PyTorch_YOLOv3), with slight modifications.


# 2. Download Weights and Install Dependencies 

For dependencies, see this file [doc/dependencies.md](doc/dependencies.md).

Download my pretrained weights file for two classes (bottle and meter) from [here](https://drive.google.com/file/d/1rGc64ks2L7OGXQceIh7GXWThglf81fTW/view?usp=sharing) and put it as `weights/yolo_trained.pth`. Then, you can run all the test cases in the next section.   


# 3. Tests
The commands for testing are listed here.

## 3.1. Synthesize images and set up YOLO files

```
bash s1_main_setup.sh             # synthesize images and set up yolo files
```

which is:
``` bash
python main_setup.py                 \
    --config_file config/config.yaml \
    --verify_mask           True     \
    --augment_imgs          True     \
    --setup_train_test_txt  True     \
    --setup_yolo            True    
```

The inputs are [config/](config) and [data/custom1/](data/custom1/); The outputs are the synthesized images and YOLO configurations files in [data/custom1_generated/](data/custom1_generated/).

## 3.2. Train
```
bash s2_train.sh                  # train yolo
```
which is:
``` bash
WEIGHTS_FILE="weights/yolo_trained.pth"
python src/train.py \
    --config_file config/config.yaml \
    --epochs 1000 \
    --learning_rate 0.0005 \
    --checkpoint_interval 20 \
    --pretrained_weights $WEIGHTS_FILE \
    --batch_size 2 
```
The trained models will be saved to: `checkpoints/$TIME_OF_TRAINING/`

## 3.3. Inference one image

```
bash s4_inference_one_image.sh    # detecting objects from an image
```

which is:
``` bash
python src/detect_one_image.py \
    --config_path "config/config.yaml" \
    --weights_path "weights/yolo_trained.pth" \
    --image_filename "test_data/images/00011.png"
```

## 3.4. Inference multiple images from webcam, folder, or video file

```
bash s3_inference_images.sh       # detecting objects from webcam, folder, or video.
```

which is:
``` bash
# -- 1. Select one of the 3 data sources by commenting out the other two

# src_data_type="webcam"
# image_data_path="none"

# src_data_type="folder"
# image_data_path="test_data/images/"

src_data_type="video"
image_data_path="test_data/video.avi"

# -- 2. Detect
python src/detect_images.py \
    --config_path "config/config.yaml" \
    --weights_path "weights/yolo_trained.pth" \
    --src_data_type $src_data_type \
    --image_data_path $image_data_path
```

## 3.5. Inference image from ROS topic

Download ROS images publisher:
```
cd ~/catkin_ws/src
git clone https://github.com/felixchenfy/ros_images_publisher
```

Publish testing images:
```
ROOT=$(rospack find ros_yolo_as_template_matching)
rosrun ros_images_publisher publish_images.py \
    --images_folder $ROOT/test_data/images/ \
    --topic_name test_data/color \
    --publish_rate 5
```

Start YOLO detection server:
```
ROOT=$(rospack find ros_yolo_as_template_matching)
rosrun ros_yolo_as_template_matching ros_server.py \
    --config_path $ROOT/config/config.yaml \
    --weights_path $ROOT/weights/yolo_trained.pth \
    --src_topic_img test_data/color \
    --dst_topic_img yolo/image \
    --dst_topic_res yolo/results # See `msg/DetectionResults.msg`
```

You can view the result on this topic `yolo/image` by tools like `rqt_image_view`.


## 4. Train Your Own Model

### 4.1. Configuration file
In [config/config.yaml](config/config.yaml), set the "data_name" and "labels" to yours.
```
data_name: "custom1" # Your data folder will be "data/custom1/"
labels: ["bottle", "meter"] # class names of the target objects
```

Set the "template_aug_effects" to meet your need.  
The other settings are also illustrated in that yaml file.

### 4.2. Training data
Create a folder "data/$data_name/", such as "data/custom1/". Create the following subfolders:
```
data/custom1
├── background
├── template_img
└── template_mask
```

### 4.2.1. template_img
Put your template images into [data/custom1/template_img/](data/custom1/template_img/) with a name of "name_index" and suffix of ".jpg" or ".png".  
```
template_img/
├── bottle_1.jpg
├── bottle_2.jpg
├── meter_1.jpg
└── meter_2.jpg
```
For example, [meter_1.jpg](data/custom1/template_img/meter_1.jpg) as shown in figure (a) below.  

![doc/fig2_data.png](doc/fig2_data.png)

### 4.2.2. template_mask
Copy the above images to [data/custom1/template_mask/](data/custom1/template_mask/). Use image editing tool to mask them as shown in figure (b) above.

Format 1: Color/Gray image, where white is the object.  
Format 2: Image with transparency channel (a 4-channel image). The non-transparent region is the object.
```
template_mask/
├── bottle_1.png
├── bottle_2.png
├── meter_1.jpg
└── meter_2.png
```
### 4.2.3. background

I downloaded 25 images from google by using [googleimagesdownload](https://github.com/hardikvasa/google-images-download) and the command:
> $ googleimagesdownload --keywords "room images" --limit 25

Copy these background images into [data/custom1/template_mask/](data/custom1/background/)

It'll be better to add the background images of your own scenes, which increases the detection precision.

## 4.3. Synthesize images and setup yolo

Run:  
> $ bash s1_main_setup.sh

This will create following staffs:
```
data/custom1_generated/
├── classes.names   # yolo
├── images/         # yolo
├── images_with_bbox/       # for your verification
├── labels/         # yolo
├── masked_template/        # for your verification
├── train.txt       # yolo
├── valid_images/           # copied images for you to test
├── valid.txt       # yolo
├── yolo.cfg        # yolo
└── yolo.data       # yolo
```

If you think the synthesized data are not enough and want to add more labeled data, you can put them into the `data/custom1_generated/images` and `data/custom1_generated/labels` folder, and then update the yolo files by:

> $ python main_setup.py --setup_yolo True

## 4.4. Train Yolo



Download the weights file from here:
```
$ cd weights/  
$ wget -c https://pjreddie.com/media/files/darknet53.conv.74
```

Put it to `weights/darknet53.conv.74`. Then, start training:

```
WEIGHTS_FILE="weights/darknet53.conv.74"
python src/train.py \
    --config_file config/config.yaml \
    --epochs 1000 \
    --learning_rate 0.0005 \
    --checkpoint_interval 20 \
    --pretrained_weights $WEIGHTS_FILE \
    --batch_size 2 
```

The checkpoints are saved to the [checkpoints/](checkpoints) folder.

# 5. Reference
https://github.com/eriklindernoren/PyTorch-YOLOv3

# 6. Trouble shooting

* ImportError: bad magic number in 'config': b'\x03\xf3\r\n'  
    $ find . -name \*.pyc -delete 