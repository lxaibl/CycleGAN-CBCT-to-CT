#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:12:29 2018

@author: s164435
"""
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


class ImagePool(object):
    def __init__(self, maxsize=50,x=512,y=512,channel=1):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = np.zeros((maxsize,x,y,channel),dtype=np.float32)

    def __call__(self, image,batch_size):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images[self.num_img:(self.num_img+batch_size)]=image
            self.num_img += batch_size
            return image
        random_choice = np.random.choice(self.images.shape[0],batch_size,replace=False)
        tmp = self.images[random_choice]
        self.images[random_choice] = image
        return tmp

def conv2d(x, name, dim, k, s, p, bn, af, is_train):
  with tf.variable_scope(name):
    w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
      initializer=tf.random_normal_initializer(stddev=0.02))
    x = tf.nn.conv2d(x, w, [1, s, s, 1], p)
    

    if bn:
      x = instance_norm(x, "bn")

    if af:
      x = leaky_relu(x)

  return x


def deconv2d(input_,name, output_shape,k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
    return deconv


def Dc_conv2d(x, name, dim, k, s, p, bn, af, is_train):
  with tf.variable_scope(name):
    w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
      initializer=tf.random_normal_initializer(stddev=0.02))
    x = tf.nn.conv2d(x, w, [1, s, s, 1], p)

    if bn:
      x = instance_norm(x, "bn")
    
    
    if af:
      x = leaky_relu(x)

  return x


def resnet(x,dim, k, is_train, name=None):
  with tf.variable_scope(name):
    with tf.variable_scope('layer1'):
      p=int((k-1)/2)
      y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
      w1 = tf.get_variable('weight1', [k, k, x.get_shape()[-1], dim],initializer=tf.random_normal_initializer(stddev=0.02))
      y = tf.nn.conv2d(y, w1,[1, 1, 1, 1], padding='VALID')
      y = instance_norm(y,"bn1" )
      y=tf.nn.relu(y)

    with tf.variable_scope('layer2'):
      y = tf.pad(y, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
      w2 = tf.get_variable("weight2",shape=[k, k, y.get_shape()[3], dim],initializer=tf.random_normal_initializer(stddev=0.02))
      y = tf.nn.conv2d(y, w2,strides=[1, 1, 1, 1], padding='VALID')
      y = instance_norm(y, "bn2")
    output = x+y
    return output  


def n_res_blocks(input, n,k, is_train):
  dim = input.get_shape()[3]
  for i in range(1,n+1):
    output = resnet(input, dim,k,is_train, 'Resnet{}_{}'.format(dim, i))
    input = output
  return output


def deconv(x, name,dim,k,bn,af,is_train):
  with tf.variable_scope(name):
    input_shape = x.get_shape().as_list()
    w = tf.get_variable("weight",[k, k, dim, input_shape[3]],initializer=tf.random_normal_initializer(stddev=0.02))
    output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, dim]
    x = tf.nn.conv2d_transpose(x, w,output_shape=output_shape,strides=[1, 2, 2, 1], padding='SAME')
    if bn:
      x = instance_norm(x,"bn")
    if af:
      x=tf.nn.relu(x)
    return x


def instance_norm(x,name):
  with tf.variable_scope(name):
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    scale = tf.get_variable('scale',[x.get_shape()[-1]],initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02,dtype=tf.float32))
    offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
    out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
    return out


def leaky_relu(input, slope=0.2):
  return tf.maximum(slope*input, input)


def mae(x,y):
  return tf.reduce_mean(tf.abs(x-y))


def mse(x, y):
  return tf.reduce_mean(tf.square(x - y))


def sce(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def mkdir(dir_path):
  try :
    os.makedirs(dir_path)
  except: pass 


def save_image_with_scale(path, arr):
  plt.imsave(path, arr,cmap='gray')
