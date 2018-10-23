import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from crypto_defense.mnist_example import deepnn
from crypto_defense.fgm import fgm
from matplotlib import pyplot as plt


checkpoint_path = '/mnt/extra_storage/checkpoints_cache/Hackathon/my-model.ckpt-900.meta'
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')

batch = mnist.test.next_batch(100, shuffle=False)
images = batch[0]
targets = batch[1]
example_image = images[3]
example_target = targets[3]

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.int64, [None])

logits, _ = deepnn(x)

adv_x = fgm(x, logits)

attacked_logits, _ = deepnn(adv_x)

softmax = tf.nn.softmax(logits)
attacked_softmax = tf.nn.softmax(attacked_logits)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path)
    att_softmax, softmax, adv_image, orig_image = sess.run((attacked_softmax, softmax, adv_x, x),
                                                           feed_dict={x: example_image})
    print("ground truth label: ", example_target)
    print("prediction on original image: ", softmax)
    print("prediction on adversarial image: ", attacked_softmax)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(adv_image)

    plt.subplot(212)
    plt.imshow(orig_image)
    plt.show()
