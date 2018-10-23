import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from crypto_defense.mnist_example import deepnn, encrypt_images
from crypto_defense.fgm import fgm
from matplotlib import pyplot as plt

patch_size = 4
if patch_size != 0:
    checkpoint_path_enc = f'/home/hack/Hackathon/checkpoints/niko_4_encrypt_w_key/my-model.ckpt-900'
    checkpoint_path = '/home/hack/Hackathon/checkpoints/my-model.ckpt-900'
    key_path = f'/home/hack/Hackathon/checkpoints/niko_4_encrypt_w_key/key.npy'
    key = np.load(key_path)
else:
    checkpoint_path = f'/home/hack/Hackathon/checkpoints/my-model.ckpt-900'

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')

batch = mnist.train.next_batch(100, shuffle=False)
images = batch[0]
targets = batch[1]
example_image = images[5]
example_target = targets[5]

# Create the model
x = tf.placeholder(tf.float32, [784])
y_ = tf.placeholder(tf.int64, [None])

if patch_size != 0:
    example_image_1 = encrypt_images([np.reshape(example_image, [1, 28, 28, 1]), example_target], key, patch_size)
logits, _ = deepnn(x)

softmax = tf.nn.softmax(logits)
adv_x = fgm(x, logits, eps=0.0000001)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path)
    softmax, orig_image = sess.run((softmax, x), feed_dict={x: example_image})
    adv_image = sess.run(adv_x, feed_dict={x: example_image})

attacked_logits, _ = deepnn(x)
attacked_softmax = tf.nn.softmax(attacked_logits)

if patch_size != 0:
    adv_image_perm = encrypt_images([np.reshape(adv_image, [1, 28, 28, 1]), example_target], key, patch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path_enc)
    attacked_softmax_res = sess.run(attacked_softmax, feed_dict={x: adv_image_perm})
    softmax_res = sess.run(attacked_softmax, feed_dict={x: example_image_1})

    print("ground truth label: ", example_target)
    print("prediction of non-P network on original image: ", softmax)
    print("prediction of P-network on permuted adversarial image: ", attacked_softmax_res)
    print("prediction of P-network on permuted original image: ", softmax_res)

    plt.figure(1)
    plt.subplot(221)
    plt.imshow(np.reshape(adv_image, [28, 28]))

    plt.subplot(222)
    plt.imshow(np.reshape(orig_image, [28, 28]))

    plt.subplot(223)
    plt.imshow(np.reshape(adv_image_perm, [28, 28]))

    plt.subplot(224)
    plt.imshow(np.reshape(example_image_1, [28, 28]))
    plt.show()
