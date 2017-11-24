'''
GoogLeNet TensorFlow Implementation.
From the paper, "Going Deeper with Convolutions", Christian Szegedy et al.

This is for the model of GooLeNet.

by Chang Jo Kim

'''

import math
import tensorflow as tf
import numpy as np

### Weight initialization
# One should generally initialize weights with a small amount of noise for symmetry breaking,
#   and to prevent 0 gradients.
# Since we're using ReLU neurons,
#   it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"


# The size of the receptive field in our network is 224x224
# in the RGB color space with zero mean

IN_SIZE = 224
NUM_CLASSES = 3
COLOR_CHANNELS = 3

def weight_variable(shape, name=None):
    # Ref. https://github.com/tflearn/tflearn/blob/master/tflearn/initializations.py
    factor = 1.0
    input_size = 1.0
    for dim in shape[:-1]:
        input_size *= float(dim)
    max_val = math.sqrt(3 / input_size) * factor
    initial = tf.random_uniform(shape, -max_val, max_val, tf.float32)
    # initial = tf.truncated_normal(shape, stddev=0.01)

    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, filter_size, stride, num_filters, name_scope=None):
    with tf.name_scope(name_scope):
        in_channels, out_channels = x.get_shape()[-1].value, num_filters
        W = weight_variable([filter_size, filter_size, in_channels, out_channels], name='weights')
        b = bias_variable([out_channels], name='biases')

        h_conv = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME") + b)

    return h_conv


def lrn(x, name_scope=None):
    with tf.name_scope(name_scope):
        h_lrn = tf.nn.lrn(x, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)

    return h_lrn


def maxpool(x, window_size, stride, name_scope=None):
    with tf.name_scope(name_scope):
        h_pool = tf.nn.max_pool(x, ksize=[1, window_size, window_size, 1], strides=[1, stride, stride, 1],
                                padding="SAME")

    return h_pool


def avgpool(x, window_size, stride, name_scope=None):
    with tf.name_scope(name_scope):
        h_pool = tf.nn.avg_pool(x, ksize=[1, window_size, window_size, 1], strides=[1, stride, stride, 1],
                                padding="VALID")

    return h_pool


def fully_connected(x, unit_size, name_scope=None):
    with tf.name_scope(name_scope):
        width, height, num_filters = x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value
        num_prev_units = width * height * num_filters
        W_fc1 = weight_variable([num_prev_units, unit_size], name='weights')
        b_fc1 = bias_variable([unit_size], name='biases')

        x_flat = tf.reshape(x, [-1, num_prev_units])
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

    return h_fc1


def dropout(x, keep_prob, name_scope=None):
    with tf.name_scope(name_scope):
        h_drop = tf.nn.dropout(x, keep_prob)

    return h_drop


def softmax(x, num_classes, name_scope=None):
    with tf.name_scope(name_scope):
        num_prev_units = np.prod([x.get_shape()[i].value for i in range(len(x.get_shape())) if i > 0])
        W = weight_variable([num_prev_units, num_classes], name='weights')
        b = bias_variable([num_classes], name='biases')

        x_flat = tf.reshape(x, [-1, num_prev_units])
        y = tf.nn.softmax(tf.matmul(x_flat, W) + b)

    return y


def loss(logits, logits_aux_1, logits_aux_2, labels, name_scope=None):
    with tf.name_scope(name_scope):
        cross_entropy_0 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                         name='xentropy_0')
        # cross_entropy_0 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy_0')
        loss_0 = tf.reduce_mean(cross_entropy_0, name='xentropy_mean_0')
        cross_entropy_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_aux_1,
                                                                         name='xentropy_aux_1')
        # cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_aux_1, name='xentropy_aux_1')
        loss_1 = tf.reduce_mean(cross_entropy_1, name='xentropy_mean_aux_1')
        cross_entropy_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_aux_2,
                                                                         name='xentropy_aux_2')
        # cross_entropy_2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_aux_2, name='xentropy_aux_2')
        loss_2 = tf.reduce_mean(cross_entropy_2, name='xentropy_mean_aux_2')

        loss = loss_0 + 0.3 * (loss_1 + loss_2)

    return loss


# h_inception3a = inception_module(h_pool2, [64, 96, 128, 16, 32, 32], name_scope='inception_3a')
def inception_module(x, nums_filters, name_scope=None):
    n_1x1_conv1, n_1x1_conv2, n_3x3_conv, n_1x1_conv3, n_5x5_conv, n_1x1_conv4 = nums_filters

    with tf.name_scope(name_scope):
        h_1x1_conv1 = conv2d(x, filter_size=1, stride=1, num_filters=n_1x1_conv1, name_scope='1x1_1S_conv1')

        h_1x1_conv2 = conv2d(x, filter_size=1, stride=1, num_filters=n_1x1_conv2, name_scope='1x1_1S_conv2')
        h_3x3_conv = conv2d(h_1x1_conv2, filter_size=3, stride=1, num_filters=n_3x3_conv, name_scope='3x3_1S_conv')

        h_1x1_conv3 = conv2d(x, filter_size=1, stride=1, num_filters=n_1x1_conv3, name_scope='1x1_1S_conv3')
        h_5x5_conv = conv2d(h_1x1_conv3, filter_size=5, stride=1, num_filters=n_5x5_conv, name_scope='5x5_1S_conv')

        h_pool = maxpool(x, window_size=3, stride=1, name_scope='MaxPool3x3_1S')
        h_1x1_conv4 = conv2d(h_pool, filter_size=1, stride=1, num_filters=n_1x1_conv4, name_scope='1x1_1S_conv4')

        h = tf.concat([h_1x1_conv1, h_3x3_conv, h_5x5_conv, h_1x1_conv4], axis=3)

    return h


def aux_network(x, keep_prob, name_scope=None):
    with tf.name_scope(name_scope):
        h_avgpool = avgpool(x, window_size=5, stride=3, name_scope='avgpool5x5_3S')
        h_1x1_conv = conv2d(h_avgpool, filter_size=1, stride=1, num_filters=128, name_scope='1x1_1S_conv')

        h_fc = fully_connected(h_1x1_conv, unit_size=1024, name_scope='fully_connected')
        h_fc_drop = dropout(h_fc, keep_prob, name_scope='dropout_70')

        y = softmax(h_fc_drop, num_classes=NUM_CLASSES, name_scope='softmax')

    return y


def model(images, keep_prob, is_training=True):
    # print(images)
    mean = tf.reduce_mean(images, axis=[1, 2, 3])
    images = images - tf.reshape(mean, shape=[-1, 1, 1, 1])
    h_conv1 = conv2d(images, filter_size=7, stride=2, num_filters=64, name_scope='Conv1')
    h_pool1 = maxpool(h_conv1, window_size=3, stride=2, name_scope='MaxPool3x3_2S_1')
    h_lrn1 = lrn(h_pool1, name_scope='LocalRespNorm1')

    h_1x1_conv1 = conv2d(h_lrn1, filter_size=1, stride=1, num_filters=64, name_scope='1x1_1S_conv1')
    h_3x3_conv1 = conv2d(h_1x1_conv1, filter_size=3, stride=1, num_filters=192, name_scope='3x3_1S_conv1')
    h_lrn2 = lrn(h_3x3_conv1, name_scope='LocalRespNorm2')

    h_pool2 = maxpool(h_lrn2, window_size=3, stride=2, name_scope='MaxPool3x3_2S_2')
    h_inception3a = inception_module(h_pool2, [64, 96, 128, 16, 32, 32], name_scope='inception_3a')
    h_inception3b = inception_module(h_inception3a, [128, 128, 192, 32, 96, 64], name_scope='inception_3b')

    h_pool3 = maxpool(h_inception3b, window_size=3, stride=2, name_scope='MaxPool3x3_2S_3')
    h_inception4a = inception_module(h_pool3, [192, 96, 208, 16, 48, 64], name_scope='inception_4a')

    h_inception4b = inception_module(h_inception4a, [160, 112, 224, 24, 64, 64], name_scope='inception_4b')
    if is_training:
        y_aux_1 = aux_network(h_inception4a, keep_prob=0.3, name_scope='inception_4a_aux')
    else:
        y_aux_1 = 0

    h_inception4c = inception_module(h_inception4b, [128, 128, 256, 24, 64, 64], name_scope='inception_4c')
    h_inception4d = inception_module(h_inception4c, [112, 144, 288, 32, 64, 64], name_scope='inception_4d')
    h_inception4e = inception_module(h_inception4d, [256, 160, 320, 32, 128, 128], name_scope='inception_4e')
    if is_training:
        y_aux_2 = aux_network(h_inception4d, keep_prob=0.3, name_scope='inception_4d_aux')
    else:
        y_aux_2 = 0

    h_pool4 = maxpool(h_inception4e, window_size=3, stride=2, name_scope='MaxPool3x3_2S_4')
    h_inception5a = inception_module(h_pool4, [256, 160, 320, 32, 128, 128], name_scope='inception_5a')
    h_inception5b = inception_module(h_inception5a, [384, 192, 384, 48, 128, 128], name_scope='inception_5b')

    h_avgpool = avgpool(h_inception5b, window_size=7, stride=1, name_scope='avgpool7x7_1S')
    h_drop = dropout(h_avgpool, keep_prob, name_scope='dropout_40')
    y = softmax(h_drop, num_classes=NUM_CLASSES, name_scope='softmax')

    return y, y_aux_1, y_aux_2


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
