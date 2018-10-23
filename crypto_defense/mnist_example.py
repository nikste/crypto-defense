# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy as np

from permute import encrypt
from p_block import random_key, p_neighbors

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([8, 8, 1, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  relu_1 = tf.nn.relu(h_conv1)

  # Second convolutional layer -- maps 64 feature maps to 128.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([6, 6, 64, 128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(relu_1, W_conv2) + b_conv2)

  relu_2 = tf.nn.relu(h_conv2)

  # Third convolutional layer -- maps 128 feature maps to 128.
  with tf.name_scope('conv2'):
    W_conv3 = weight_variable([6, 6, 128, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(relu_2, W_conv3) + b_conv3)

  relu_3 = tf.nn.relu(h_conv3)

  flatten = tf.manip.reshape(relu_3, [-1, 28 * 28 * 128])

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([28 * 28 * 128, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(flatten, W_fc2) + b_fc2
  return y_conv, 0


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def encrypt_images(batch: np.array, key:np.array, patch_size=2) -> np.array:
  img, labels = batch
  img_reshape = np.reshape(img, [img.shape[0], 28, 28, 1])
  list_array = np.split(img_reshape, indices_or_sections=img.shape[0], axis=0)
  #img_reshape = np.array([p_neighbors(np.squeeze(x, axis=0), key, patch_size=patch_size) for x in list_array])
  img_reshape = np.array([encrypt(np.squeeze(x, axis=0)) for x in list_array])
  batch = (np.reshape(img_reshape, [50, 784]), labels)
  return batch

def main(_):
  encrypt = FLAGS.encrypt
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  saver = tf.train.Saver()

  if encrypt:
    print("USING ENCRYPTION!")
    key = random_key(np.zeros([28,28,3]))
  t_start = time.time()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
      batch = mnist.train.next_batch(50)
      if encrypt:
        batch = encrypt_images(batch, key, encrypt)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print("step", i, " training accuracy", train_accuracy, "took", time.time() - t_start)
        t_start = time.time()
        saver.save(sess=sess, save_path=f"{graph_location}/my-model.ckpt", global_step=i)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # compute in batches to avoid OOM on GPUs
    accuracy_l = []
    for _ in range(20):
      batch = mnist.test.next_batch(500, shuffle=False)
      if encrypt:
        batch = encrypt_images(batch, key, patch_size=encrypt)
      accuracy_l.append(accuracy.eval(feed_dict={x: batch[0],
                                                 y_: batch[1]}))
    print('test accuracy %g' % np.mean(accuracy_l))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--encrypt', type=int,
                      default=2,
                      help='use encryption for training and inference, specify an integer value for patchsize in pixel,'
                           'set to 0 to disable')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

