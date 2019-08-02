"""
[WIP] Contains VAE decoders for GQN.
Currently under development, use at yor own risk!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .gqn_params import GQN_DEFAULT_CONFIG
from .gqn_utils import broadcast_pose


def vae_simple_encoder(x, scope="VAESimpleEncoder"):
  with tf.compat.v1.variable_scope(scope):
    endpoints = {}

    net = x  # shape (b, 64, 64, 3)
    net = tf.compat.v1.layers.conv2d(
      net, kernel_size=3, filters=512, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 64, 64, 512)
    net = tf.compat.v1.layers.conv2d(
      net, kernel_size=3, filters=64, strides=2, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 32, 32, 64)
    net = tf.compat.v1.layers.conv2d(
      net, kernel_size=3, filters=128, strides=2, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 16, 16, 128)
    net = tf.compat.v1.layers.conv2d(
      net, kernel_size=5, filters=512, strides=2, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 8, 8, 512)

    # to go from (8, 8, 512) to (1, 1, 256), we do a regular strided convolution
    # and move the extra spatial information to channels
    net = tf.compat.v1.layers.conv2d(
      net, kernel_size=5, filters=16, strides=2, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 4, 4, 16)
    net = tf.compat.v1.space_to_depth(input=net, block_size=4)  # shape out: (b, 1, 1, 256)

    return net, endpoints


def vae_simple_decoder(z, scope="VAESimpleDecoder"):
  def _upsample_conv2d(net, factor, filters, **kwargs):
    net = tf.compat.v1.layers.conv2d(net, filters=factor*factor*filters, **kwargs)
    net = tf.compat.v1.depth_to_space(input=net, block_size=factor)

    return net

  with tf.compat.v1.variable_scope(scope):
    endpoints = {}

    net = z  # shape (b, 1, 1, c)
    net = _upsample_conv2d(
      net, kernel_size=3, filters=128, factor=16, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 16, 16, 128)
    net = _upsample_conv2d(
      net, kernel_size=3, filters=512, factor=2, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 32, 32, 512)
    net = _upsample_conv2d(
      net, kernel_size=3, filters=512, factor=2, activation=tf.nn.relu,
      padding="SAME")  # shape out: (b, 64, 64, 512)
    net = tf.compat.v1.layers.conv2d(net, kernel_size=3, filters=3, padding="SAME")

    return net, endpoints


def vae_tower_decoder(
    z, query_pose, output_channels=GQN_DEFAULT_CONFIG.LSTM_CANVAS_CHANNELS,
    scope="VAETowerDecoder"):
  """
  Defines VAE tower decoder graph for image generation based on a GQN scene \
  encoding.
  """
  with tf.compat.v1.variable_scope(scope):
    endpoints = {}

    net = z
    net = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = tf.compat.v1.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    height, width = tf.shape(input=net)[1], tf.shape(input=net)[2]
    query_pose = broadcast_pose(query_pose, height, width)
    net = tf.concat([net, query_pose], axis=-1)

    skip1 = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=1, strides=1,
                             padding="SAME", activation=None)

    net = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = net + skip1

    net = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)

    net = tf.image.resize(net, size=(2 * height, 2 * width), method=tf.image.ResizeMethod.BILINEAR)

    net = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    skip2 = tf.compat.v1.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.compat.v1.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = net + skip2

    net = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = tf.image.resize(net, size=(2 * height, 2 * width), method=tf.image.ResizeMethod.BILINEAR)

    net = tf.compat.v1.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = tf.compat.v1.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = tf.compat.v1.layers.conv2d(net, filters=output_channels, kernel_size=1,
                           strides=1, padding="SAME", activation=None)

    return net, endpoints
