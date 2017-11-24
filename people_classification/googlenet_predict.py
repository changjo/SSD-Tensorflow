'''
Predict an image label with a trained model

  by Chang Jo Kim
'''

import tensorflow as tf
import googlenet
import data_utils
import scipy
import skimage.transform
import numpy as np
import os
import visualization

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## Please install
# scipy, numpy, pillow

tf.app.flags.DEFINE_string('checkpoints', 'checkpoints/model.ckpt-39999', 'A pre-trained checkpoint file.')
FLAGS = tf.app.flags.FLAGS

CLASS2LABEL = {0: "Unknown", 1: "Boeun", 2: "Chang Jo"}

def tensorflow_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    session = tf.Session(config=config)

    return session


def restore_checkpoints(session, ckpt_filename):
    # saver = tf.train.import_meta_graph(ckpt_filename + '.meta')
    saver = tf.train.Saver()
    saver.restore(session, ckpt_filename)


def image_preprocess(image):
    # image = scipy.misc.imresize(image, (googlenet.IN_SIZE, googlenet.IN_SIZE))
    image = skimage.transform.resize(image, (googlenet.IN_SIZE, googlenet.IN_SIZE))
    return image

def images_preprocess(images):
    # image = scipy.misc.imresize(image, (googlenet.IN_SIZE, googlenet.IN_SIZE))
    new_images = []
    for image in images:
        new_image = image_preprocess(image)
        new_images.append(new_image)
    return np.array(new_images)


def start():
    # images, labels = data_utils.inputs(FLAGS.tfrecord, BATCH_SIZE, num_epochs=None)
    # x = tf.placeholder(tf.float32, shape=[None, 784])
    x = tf.placeholder(tf.float32, shape=[None, googlenet.IN_SIZE, googlenet.IN_SIZE, googlenet.COLOR_CHANNELS])
    keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')

    logits, logits_aux_1, logits_aux_2 = googlenet.model(x, keep_prob_placeholder, is_training=False)

    session = tensorflow_session()
    session.run(tf.global_variables_initializer())

    restore_checkpoints(session, FLAGS.checkpoints)

    def predict(images):

        # images = np.reshape(image, (1, googlenet.IN_SIZE, googlenet.IN_SIZE, googlenet.COLOR_CHANNELS))
        # images = images.astype(np.float32) * (1. / 255)
        # mean = np.mean(image, axis=(1, 2, 3))
        # image = image - np.reshape(mean, (1, 1, 1, 1))
        logits_values = session.run(logits, feed_dict={x: images, keep_prob_placeholder: 1.0})
        # print(logits_values)
        predicted_labels = session.run(tf.argmax(logits_values, 1))

        return predicted_labels

    return predict


def main(_):

    # class2label = {0: "Boeun", 1: "Chang Jo"}

    # Start and get a predict function
    predict = start()

    # image_files = ["data/images/k00000010.jpg", "data/images/k00000102.jpg", "data/friday.jpg"]
    images_path = 'data/images/'
    lables_path = 'data/labels/'
    image_files = [os.path.join(images_path, image_filename) for image_filename in sorted(os.listdir(images_path))]

    # Perform prediction
    predicted_classes = []
    true_count = 0
    for image_file in image_files:
        image, shape = data_utils.get_image(image_file)
        image = image_preprocess(image)
        
        # import cv2
        # cv2.imshow('frame', image)        
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        predicted_classes.append(predict(np.array([image])))
        head, tail = os.path.splitext(image_file.split('/')[-1])
        true_class = data_utils.get_label_int(os.path.join(lables_path, head + '.txt'))

        print(image_file, "->", predicted_classes[-1], predicted_classes[-1] == true_class)
        if predicted_classes[-1] == true_class:
            true_count += 1

    print("Accuracy: %.2f" % (float(true_count) / len(image_files)))
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tf.app.run()