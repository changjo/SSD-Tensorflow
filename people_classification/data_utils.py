""" Data Utils

Reference:
https://github.com/changjo/stanford-tensorflow-tutorials/blob/master/examples/09_tfrecord_example.py
https://github.com/balancap/SSD-Tensorflow/blob/master/datasets/pascalvoc_to_tfrecords.py
"""

from PIL import Image
import numpy as np
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import random
import os, sys

DIRECTORY_IMAGES = 'images/'
DIRECTORY_LABELS = 'labels/'

# TFRecords 변환 파라미터.
RANDOM_SEED = 4242
SAMPLES_PER_FILE = 200
COLOR_CHANNELS = 3
IN_SIZE = 224


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_image(filename):
    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    height, width, depth = shape[0], shape[1], shape[2]
    shape = (height, width, depth)

    return image, shape


def get_label_int(filename):
    if not os.path.isfile(filename):
        return -1
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) != 0:
                label = int(line)
                return label


def _convert_to_example(label, shape, binary_image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(binary_image),
        'label': _int64_feature(label),
        'height': _int64_feature(shape[0]),
        'width': _int64_feature(shape[1]),
        'depth': _int64_feature(shape[2])
    }))
    return example


def _add_to_tfrecord(dataset_dir, filename, tfrecord_writer):
    name, ext = os.path.splitext(filename)
    image_file = os.path.join(dataset_dir, DIRECTORY_IMAGES, name + ext)
    image, shape = get_image(image_file)
    image = scipy.misc.imresize(image, (IN_SIZE, IN_SIZE))
    binary_image = image.tobytes()

    label_file = os.path.join(dataset_dir, DIRECTORY_LABELS, name + '.txt')
    label = get_label_int(label_file)

    example = _convert_to_example(label, shape, binary_image)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def convert_to_tfrecord(dataset_dir, output_dir, name="output", shuffling=False):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    path = os.path.join(dataset_dir, DIRECTORY_LABELS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    tfrecord_idx = 0
    while i < len(filenames):
        tfrecord_file = _get_output_filename(output_dir, name, tfrecord_idx)
        with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
            j = 0
            while j < SAMPLES_PER_FILE and i < len(filenames):
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))

                filename = filenames[i]
                _add_to_tfrecord(dataset_dir, filename, writer)

                i += 1
                j += 1
            tfrecord_idx += 1


def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'image': tf.FixedLenFeature([], tf.string),
                                                    'label': tf.FixedLenFeature([], tf.int64),
                                                    'height': tf.FixedLenFeature([], tf.int64),
                                                    'width': tf.FixedLenFeature([], tf.int64),
                                                    'depth': tf.FixedLenFeature([], tf.int64)
                                                }, name='features')

    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    label = tf.cast(tfrecord_features['label'], tf.int32)
    height = tf.cast(tfrecord_features['height'], tf.int32)
    width = tf.cast(tfrecord_features['width'], tf.int32)
    depth = tf.cast(tfrecord_features['depth'], tf.int32)
    # the image tensor is flatterned out, so we have to reconstruct the shape.
    image = tf.reshape(image, [height, width, depth])

    return image, label, height, width, depth


def read_and_decode(tfrecord_file_queue):
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'image': tf.FixedLenFeature([], tf.string),
                                                    'label': tf.FixedLenFeature([], tf.int64),
                                                    'height': tf.FixedLenFeature([], tf.int64),
                                                    'width': tf.FixedLenFeature([], tf.int64),
                                                    'depth': tf.FixedLenFeature([], tf.int64)
                                                }, name='features')

    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    label = tf.cast(tfrecord_features['label'], tf.int32)
    height = tf.cast(tfrecord_features['height'], tf.int32)
    width = tf.cast(tfrecord_features['width'], tf.int32)
    depth = tf.cast(tfrecord_features['depth'], tf.int32)
    # the image tensor is flatterned out, so we have to reconstruct the shape.
    image = tf.cast(image, tf.float32) * (1. / 255)
    shape = [height, width, COLOR_CHANNELS]
    image = tf.reshape(image, shape)

    return image, label, shape


def read_tfrecord(tfrecord_file):
    image, label, height, width, depth = read_from_tfrecord([tfrecord_file])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # label, image, shape = sess.run([label, image, shape])
        im, la, h, w, d = sess.run([image, label, height, width, depth])
        coord.request_stop()
        coord.join(threads)

    print(la)
    print(im)
    print(h, w, d)
    plt.imshow(im)
    plt.show()


# From "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py"
def inputs_old(filename, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):
        tfrecord_file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs, name='queue')

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label, shape = read_and_decode(tfrecord_file_queue)
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        # images, sparse_labels = tf.train.shuffle_batch(
        #     [image, label], batch_size=batch_size, num_threads=2,
        #     capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
        image = image_preprocess(image)
        # image = tf.image.resize_images(image, [googlenet.IN_SIZE, googlenet.IN_SIZE])
        # image = tf.random_crop(image, [googlenet.IN_SIZE, googlenet.IN_SIZE, 3])

        # Flatten an image.
        #image = tf.reshape(image, [googlenet.IN_SIZE, googlenet.IN_SIZE, COLOR_CHANNELS])
        images, sparse_labels = tf.train.batch([image, label], batch_size=batch_size, num_threads=2,
                                               capacity=1000 + 3 * batch_size)

        return images, sparse_labels


def inputs(filenames, batch_size, num_epochs):

    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {'image': tf.FixedLenFeature((), tf.string),
                            'label': tf.FixedLenFeature((), tf.int64),
                            'height': tf.FixedLenFeature((), tf.int64),
                            'width': tf.FixedLenFeature((), tf.int64),
                            'depth': tf.FixedLenFeature((), tf.int64)
                            }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255)
        height = tf.cast(parsed['height'], tf.int32)
        width = tf.cast(parsed['width'], tf.int32)
        depth = tf.cast(parsed['depth'], tf.int32)
        image = tf.reshape(image, [IN_SIZE, IN_SIZE, COLOR_CHANNELS])

        #image = image_preprocess(image)
        label = tf.cast(parsed['label'], tf.int32)

        return image, label

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    images, labels = iterator.get_next()

    return images, labels


def image_preprocess(images):
    images = tf.image.resize_images(images, [IN_SIZE, IN_SIZE])
    return images


def main():
    dataset_dir = 'data/'
    output_dir = './tfrecords'
    name = 'our'
    # convert_to_tfrecord(dataset_dir=dataset_dir, output_dir=output_dir, name=name, shuffling=False)
    tfrecord_file = os.path.join(output_dir, name + '_000.tfrecord')
    # read_tfrecord(tfrecord_file)
    images, labels = inputs([tfrecord_file], batch_size=2, num_epochs=None)

    with tf.Session() as sess:
        im, la = sess.run([images, labels])

    print(la[0])
    print(im[0])
    plt.imshow(im[0])
    plt.show()


if __name__ == '__main__':
    main()
