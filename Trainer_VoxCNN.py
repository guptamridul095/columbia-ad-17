#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:00:32 2019

@author: mridulg
"""
import tensorflow as tf
import argparse
from google.oauth2 import service_account
from google.cloud import storage
import os

def _cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features, [-1, 256, 256, 256, 1])
  conv1 = tf.layers.conv3d(
    inputs=input_layer,
    filters=8,
    kernel_size=[3,3,3],
    strides=[2, 2, 2],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  conv2 = tf.layers.conv3d(
    inputs=conv1,
    filters=8,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
  conv3 = tf.layers.conv3d(
    inputs=pool1,
    filters=16,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  conv4 = tf.layers.conv3d(
    inputs=conv3,
    filters=16,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  pool2 = tf.layers.max_pooling3d(inputs=conv4 , pool_size=[2,2,2], strides=2)
  
  conv5 = tf.layers.conv3d(
    inputs=pool2,
    filters=32
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  conv6 = tf.layers.conv3d(
    inputs=conv5,
    filters=32,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  conv7 = tf.layers.conv3d(
    inputs=conv6,
    filters=32,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  pool3 = tf.layers.max_pooling3d(inputs=conv7 , pool_size=[2,2,2], strides=2)
  conv8 = tf.layers.conv3d(
    inputs=conv6,
    filters=64,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  conv9 = tf.layers.conv3d(
    inputs=conv8,
    filters=64,
    kernel_size=[3,3,3],
    padding='same',
    activation=tf.nn.leaky_relu
    )
  pool1_flat = tf.layers.flatten(conv8)
  dense1 = tf.layers.dense(inputs=pool1_flat, units=128, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense2, units=64, activation=tf.nn.relu)
  logits = tf.layers.dense(inputs=dense2, units=6, activation=tf.nn.softmax,
    kernel_initializer=tf.random_normal_initializer(stddev=.01))
