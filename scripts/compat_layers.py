"""Minimal replacement for the eight contrib fully_connected calls used by Decima.

Preserves dense widths, activation, zero biases, Glorot-uniform initialization,
and weights/biases scope names. Not a general tf.contrib implementation.
"""
import tensorflow.compat.v1 as tf


def fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu):
    width = inputs.shape.as_list()[-1]
    if width is None:
        raise ValueError('Decima dense input width must be statically known')
    with tf.variable_scope(None, default_name='fully_connected'):
        weights = tf.get_variable('weights', [width, num_outputs],
                                  initializer=tf.glorot_uniform_initializer())
        biases = tf.get_variable('biases', [num_outputs], initializer=tf.zeros_initializer())
        result = tf.matmul(inputs, weights) + biases
        return result if activation_fn is None else activation_fn(result)
