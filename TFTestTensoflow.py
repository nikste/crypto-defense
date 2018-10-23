# TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batchSize = 1000

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainData = mnist.train.images
trainLabels = mnist.train.labels

filter_weights64 = tf.get_variable(shape=[8, 8, 1, 64], name="w64")
filter_weights128 = tf.get_variable(shape=[6, 6, 64, 128], name="w128")
filter_weights1282 = tf.get_variable(shape=[6, 6, 128, 128], name="w1282")

inputData = tf.placeholder(tf.float32)
inputLabels = tf.placeholder(tf.int8)

conv2d64 = tf.nn.conv2d(inputData, filter_weights64, [1, 1, 1, 1], "SAME", True)
convRelu64 = tf.nn.relu(conv2d64)

conv2d128 = tf.nn.conv2d(convRelu64, filter_weights128, [1, 1, 1, 1], "SAME", True)
convRelu128 = tf.nn.relu(conv2d128)

conv2d1282 = tf.nn.conv2d(convRelu128, filter_weights1282, [1, 1, 1, 1], "SAME", True)
convRelu1282 = tf.nn.relu(conv2d1282)


flatten = tf.manip.reshape(convRelu1282, [batchSize, 28*28*128])
logits = tf.layers.dense(flatten, 10)

softmax = tf.nn.softmax(logits)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=inputLabels, logits=logits)
train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)

saver = tf.train.Saver()

for i in range(0, len(trainData), batchSize):
    bätch = trainData[i:i+batchSize]
    inputTensor = np.reshape(bätch, [batchSize, 28, 28, 1])
    bötch = trainLabels[i:i+batchSize]
    labelTensor = np.reshape(bötch, [batchSize, 10]) # len(labels) weil 1 hot encoded
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _, softmax_res = sess.run([train_op, softmax], feed_dict={inputData: inputTensor / 255, inputLabels: labelTensor})
        print(softmax_res[0])

saver.save(sess, 'my-model')




