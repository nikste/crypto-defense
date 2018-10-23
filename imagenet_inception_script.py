import tensorflow as tf
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
import numpy as np
from .iterative_gradient_attack import iterative_gradient_attack
from .inception_model import inception


image = tf.Variable(tf.zeros((299, 299, 3)))
logits, probs = inception(image, reuse=False)
x = tf.placeholder(tf.float32, (299, 299, 3))
learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())
epsilon = tf.placeholder(tf.float32, ())

# Next, we load pre-trained weights. This Inception v3 has a top-5 accuracy of 93.9%.

data_dir = tempfile.mkdtemp()
inception_tarball, _ = urlretrieve(
    'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)

with tf.Session() as sess:
    saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))


# ## Execution

demo_epsilon = 2.0 / 255.0  # a really small perturbation
demo_lr = 1e-1
demo_steps = 100
demo_target = 924  # "guacamole"

# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i + 1) % 10 == 0:
        print('step %d, loss=%g' % (i + 1, loss_value))

adv = x_hat.eval()  # retrieve the adversarial example

# This adversarial image is visually indistinguishable from the original, with no visual artifacts. However, it's classified as "guacamole" with high probability!

# In[18]:


classify(adv, correct_class=img_class, target_class=demo_target)
