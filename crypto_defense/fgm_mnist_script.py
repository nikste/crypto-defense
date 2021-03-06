import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from crypto_defense.mnist_example import deepnn, encrypt_images
from crypto_defense.fgm import fgm
from matplotlib import pyplot as plt

patch_size = 0
if patch_size != 0:
    checkpoint_path = f'/home/hack/Hackathon/checkpoints/niko_{patch_size}_encrypt/my-model.ckpt-900'
else:
    checkpoint_path = f'/home/hack/Hackathon/checkpoints/my-model.ckpt-900'

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')

batch = mnist.test.next_batch(100, shuffle=False)
images = batch[0]
targets = batch[1]
example_image = images[3]
example_target = targets[3]

# Create the model
x = tf.placeholder(tf.float32, [784])
y_ = tf.placeholder(tf.int64, [None])

if patch_size != 0:
    example_image = encrypt_images(example_image, key, patch_size=patch_size)
logits, _ = deepnn(x)

softmax = tf.nn.softmax(logits)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path)
    softmax, orig_image = sess.run((softmax, x), feed_dict={x: example_image})

adv_x = fgm(x, logits, eps=0.000001)
attacked_logits, _ = deepnn(x)
attacked_softmax = tf.nn.softmax(attacked_logits)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path)
    adv_image = sess.run(adv_x, feed_dict={x: example_image})
    attacked_softmax_res = sess.run(attacked_softmax, feed_dict={x: adv_image})

    print("ground truth label: ", example_target)
    print("prediction on original image: ", softmax)
    print("prediction on adversarial image: ", attacked_softmax_res)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(np.reshape(adv_image, [28, 28]))

    plt.subplot(212)
    plt.imshow(np.reshape(orig_image, [28, 28]))
    plt.show()
