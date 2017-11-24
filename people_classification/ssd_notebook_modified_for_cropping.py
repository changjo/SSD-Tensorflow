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

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


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

class2label = {}

for key in VOC_LABELS.keys():
    class2label[VOC_LABELS[key][0]] = key

USE_WEBCAM = False
CROP = True

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
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = frame
        
        rclasses, rscores, rbboxes =  process_image(img)
        
        '''
        for i in range(rclasses.shape[0]):
            cls_id = int(rclasses[i])
            if cls_id >= 0:
                if cls_id not in colors:
                    colors[cls_id] = (int(random.randint(0, 255)), random.randint(0, 255), random.randint(0, 255))
        '''
        
        visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau, class2label, thickness=2)


        # Put text on the frame
        caption = u"Hello"
        location = (100, 100)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1
        color = (0, 0, 255)
        #cv2.putText(img, caption, location, font, scale, color, 1)

        #cv2.addText(img, caption, location, font)

        # Display the resulting frame
        if cnt == 0:
            cv2.imshow('frame', img)
            cnt = 1

        key = cv2.waitKey(10) & 0xFF
        if key == ord('r'):
            cv2.imshow('frame', img)

        if key == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


else:
    #img = cv2.imread('messi5.jpg',0)

    path = '/media/airc/HDD_2TB/data/keti_people/2/'
    image_names = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    print(image_names)
    #image_name = path + image_names[-2]
    # image_name = '../demo/flowers-248822_640.jpg'
    
    i = 0
    while i < len(image_names):
        image_name = image_names[i]
        head, ext = os.path.splitext(image_name)
        img = cv2.imread(os.path.join(path, image_name), 1)
        #img = cv2.imread('police-car-406893_1920.jpg', 1)
        rclasses, rscores, rbboxes =  process_image(img)

        print(image_name)
        print(rclasses)
        # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

        if CROP:
            imgs = visualization.crop_img(img, rclasses, rbboxes, crop_class=15)
        else:
            visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_tableau, class2label, thickness=2)

        #print(rbboxes)
        # # Put text on the frame
        # caption = u"Hello"
        # location = (100, 100)
        # scale = 1
        # color = (0, 0, 255)
        # font = QFont()
        # font.setFamily('Times')
        # font.setPointSize(10)

        # #cv2.putText(img, caption, location, font, scale, color, 1)

        # cv2.addText(img, caption, location, font)

        ## For Displaying Korean
        # from PIL import Image, ImageDraw, ImageFont

        # font_size = 36
        # font_color = (0, 0, 0)
        # caption = u' '
        # unicode_font = ImageFont.truetype("NanumGothic.ttf", font_size)

        # pil_image = Image.open(image_name).convert('RGB')
        # draw = ImageDraw.Draw(pil_image)
        # draw.text ( (10,10), caption, font=unicode_font, fill=font_color )
        # open_cv_image = np.array(pil_image)
        # open_cv_image = open_cv_image[:, :, ::-1].copy()

        # visualization.bboxes_draw_on_img(open_cv_image, rclasses, rscores, rbboxes, visualization.colors_tableau, class2label, thickness=2)
        ##

        # cv2.imshow('image', open_cv_image)

        if CROP:
            j = 0
            while j < len(imgs):
                #cv2.imshow('image', imgs[j])
                crop_image_name = os.path.join(os.path.join(path, 'crop'), head + '_crop_' + str(j) + ext)
                cv2.imwrite(crop_image_name, imgs[j])
                j += 1
                # if cv2.waitKey(0) == ord('s'):
                #     crop_image_name = os.path.join(os.path.join(path, 'crop'), head + '_crop' + ext)
                #     cv2.imwrite(crop_image_name, imgs[j])
                #     j += 1
                # elif cv2.waitKey(0) == ord('n'):
                #     j += 1
                # elif cv2.waitKey(0) & 0xFF == ord('q'):
                #     break
            i += 1
        else:
            if img is not None:
                cv2.imshow('image', img)
            if img is None or cv2.waitKey(0) == ord('n'):
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

