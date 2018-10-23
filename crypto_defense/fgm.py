import tensorflow as tf
import numpy as np


def fgm(x,
        logits,
        y=None,
        eps=0.000001,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False,
        sanity_checks=True):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param logits: output of model.get_logits
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

    if clip_max is not None:
        asserts.append(tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

    # Make sure the caller has not passed probs by accident
    assert logits.op.type != 'Softmax'

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(logits, 1, keepdims=True)
        y = tf.to_float(tf.equal(logits, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(range(1, len(x.get_shape())))
        avoid_zero_div = 1e-12
        avoid_nan_norm = tf.maximum(avoid_zero_div,
                                    tf.reduce_sum(tf.abs(grad),
                                                  reduction_indices=red_ind,
                                                  keepdims=True))
        normalized_grad = grad / avoid_nan_norm
    elif ord == 2:
        red_ind = list(range(1, len(x.get_shape())))
        avoid_zero_div = 1e-12
        square = tf.maximum(avoid_zero_div,
                            tf.reduce_sum(tf.square(grad),
                                          reduction_indices=red_ind,
                                          keepdims=True))
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)


    if sanity_checks:
        with tf.control_dependencies(asserts):
            adv_x = tf.identity(adv_x)

    return adv_x
