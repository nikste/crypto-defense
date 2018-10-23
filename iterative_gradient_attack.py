import tensorflow as tf
import numpy as np


def iterative_gradient_attack(image: np.ndarray,
                              logits: tf.Tensor,
                              x: tf.Tensor,
                              y_hat: tf.Tensor,
                              learning_rate: tf.Tensor,
                              epsilon: tf.Tensor):

    x_hat = image  # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)

    # Next, we write the gradient descent step to maximize the log probability of the target class
    # (or equivalently, minimize the [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)).

    labels = tf.one_hot(y_hat, 1000)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
    optim_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, var_list=[x_hat])

    # ## Projection step
    # Finally, we write the projection step to keep our adversarial example visually close to the original image.
    # Additionally, we clip to $[0, 1]$ to keep it a valid image.

    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    return assign_op, optim_step, project_step
