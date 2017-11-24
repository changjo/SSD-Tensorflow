# coding: utf-8

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.transform

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
import visualization

import googlenet_predict

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# ## SSD 300 Model
# 
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
# 
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)


# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# ## Post-processing pipeline
# 
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
# 
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Added ----------------------------------------------------------------------

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

def check_person_in_classes(classes):
    for a_class in classes:
        if VOC_LABELS['person'][0] in classes:
            return True
    return False

def replace_person_class_to_specific_person_class(classes, person_classes):

    person_class_indices = []
    for i in range(len(classes)):
        if classes[i] == 15:
            person_class_indices.append(i)
    
    for i in range(len(person_class_indices)):
        classes[person_class_indices[i]] = 21 + person_classes[i]

person_class2label = googlenet_predict.CLASS2LABEL

class2label = {}
for key in VOC_LABELS.keys():
    class2label[VOC_LABELS[key][0]] = key

# Replace 'person' with person class2label
for key in person_class2label.keys():
    class2label[21 + key] = person_class2label[key]

print(class2label)

USE_WEBCAM = True

googlenet_graph = tf.Graph()
with googlenet_graph.as_default():
    person_predict = googlenet_predict.start()

if USE_WEBCAM:

    cap = cv2.VideoCapture(0)
    print(cap.get(3), cap.get(4))
    #cap.set(3, 1920)
    #cap.set(4, 1080)
    cap.set(3, 640)
    cap.set(4, 480)
    print(cap.get(3), cap.get(4))

    colors = dict()
    cnt = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        rclasses, rscores, rbboxes =  process_image(img)
        
        if check_person_in_classes(rclasses):
            cropped_imgs = visualization.crop_img(img, rclasses, rbboxes, crop_class=VOC_LABELS['person'][0])
            preprocessed_imgs = googlenet_predict.images_preprocess(cropped_imgs)
            with googlenet_graph.as_default():
                predicted_person_classes = person_predict(preprocessed_imgs)

            replace_person_class_to_specific_person_class(rclasses, predicted_person_classes)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau, class2label, thickness=2)

        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau, class2label, thickness=2)

        # Display the resulting frame
        cv2.imshow('frame', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


else:
    #img = cv2.imread('messi5.jpg',0)

    path = '/media/airc/HDD_2TB/data/keti_people/1/'
    image_names = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    print(image_names)
    #image_name = path + image_names[-2]
    # image_name = '../demo/flowers-248822_640.jpg'
    
    i = 0
    while i < len(image_names):
        image_name = image_names[i]
        head, ext = os.path.splitext(image_name)
        img = cv2.imread(os.path.join(path, image_name), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.imread('police-car-406893_1920.jpg', 1)
        rclasses, rscores, rbboxes =  process_image(img)

        if check_person_in_classes(rclasses):
            imgs = visualization.crop_img(img, rclasses, rbboxes, crop_class=VOC_LABELS['person'][0])
            imgs = googlenet_predict.images_preprocess(imgs)
            with googlenet_graph.as_default():
                predicted_person_classes = person_predict(imgs)

            replace_person_class_to_specific_person_class(rclasses, predicted_person_classes)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau, class2label, thickness=2)

        cv2.imshow('frame', skimage.transform.rescale(img, 0.4))
        if cv2.waitKey(0) == ord('n'):
            i += 1
        elif cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
# ----------------------------------------------------------------------------




'''
# Test on some demo image and visualize output.
path = '../demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-5])
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
'''

