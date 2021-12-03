import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
import numpy as np

from layer_helpers import *

_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def generator_fn(noise, mode, weight_decay=2.5e-5):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  
    net = _dense(noise, 8192, weight_decay)
    net = _batch_norm(net, is_training)
    net = tf.nn.relu(net)

    # go from noise to a 128x128x32 image 
    net = tf.reshape(net, [-1, 4, 4, 512])
    net = _deconv2d(net, 256, 4, 2, weight_decay)
    net = _deconv2d(net, 128, 4, 2, weight_decay)
    net = _deconv2d(net, 64, 4, 2, weight_decay)
    net = _deconv2d(net, 32, 4, 2, weight_decay)
    net = _deconv2d(net, 16, 4, 2, weight_decay)

    # Reduce from 128x128x32 to 128x128x3 (original image format)
    net = _deconv2d(net, 3, 4, 1, weight_decay)
    # range should be [-1,1]. When I pass the images in, they'll be normalized
    # to [-1,1] instead of [0,255] for each RGB channel.
    net = tf.tanh(net)

    return net

def unconditional_discriminator(img, unused_conditioning, mode, weight_decay=2.5e-5):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    net = _conv2d(img, 32, 4, 2, weight_decay)
    net = _leaky_relu(net)
    
    net = _conv2d(net, 64, 4, 2, weight_decay)
    net = _leaky_relu(net)
    
    net = _conv2d(net, 128, 4, 2, weight_decay)
    net = _leaky_relu(net)
    
    net = _conv2d(net, 256, 4, 2, weight_decay)
    net = _leaky_relu(net)
    
    net = _conv2d(net, 512, 4, 2, weight_decay)
    net = _leaky_relu(net)
    
    net = tf.layers.flatten(net)
    
    net = _dense(net, 8192, weight_decay)
    net = _batch_norm(net, is_training)
    net = _leaky_relu(net)
    
    net = _dense(net, 1, weight_decay)

    return net

def gen_opt():
    gstep = tf.train.get_or_create_global_step()
    base_lr = generator_lr
    # Halve the learning rate at 1000 steps.
    lr = tf.cond(gstep < 1000, lambda: base_lr, lambda: base_lr / 2.0)
    return tf.train.AdamOptimizer(lr, 0.5)
