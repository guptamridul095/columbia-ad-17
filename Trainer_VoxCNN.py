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
  
  tf.logging.info(logits)
  #logits = tf.print(logits, [logits], message="logits: ")
  one_hot_labels = tf.one_hot(labels, 6)
  tf.logging.info(tf.shape(one_hot_labels))
  #why is loss so large? may be causing loss -> NaN
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels))
  tf.logging.info(tf.shape(loss))
  host_call=None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    batches_per_epoch = 7943/BATCH_SIZE
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.01, global_step=global_step,
        decay_steps=100, decay_rate=0.001)
    #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    optimizer = 
    ##gvs = optimizer.compute_gradients(loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    global_step = tf.train.get_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      #train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
      train_op = optimizer.minimize(loss, global_step=global_step)

    def host_call_fn(gs, loss, lr, ce):
      """Training host call. Creates scalar summaries for training metrics.
      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the
      model to the `metric_fn`, provide as part of the `host_call`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.
      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `host_call`.
      Args:
        gs: `Tensor with shape `[batch]` for the global_step
        loss: `Tensor` with shape `[batch]` for the training loss.
        lr: `Tensor` with shape `[batch]` for the learning_rate.
        ce: `Tensor` with shape `[batch]` for the current_epoch.
      Returns:
        List of summary ops to run on the CPU host.
      """
      gs = gs[0]
      with summary.create_file_writer(MODEL_DIR).as_default():
        with summary.always_record_summaries():
          summary.scalar('loss', loss[0], step=gs)
          summary.scalar('learning_rate', lr[0], step=gs)
          return summary.all_summary_ops()

    gs_t = tf.reshape(global_step, [1])
    loss_t = tf.reshape(loss, [1])
  else:
    train_op = None

  eval_metrics = None
  #if mode == tf.estimator.ModeKeys.EVAL or :
  def metric_fn(labels, logits):
    predictions = tf.argmax(logits, axis=1)

    return {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predictions)
    }

  eval_metrics = metric_fn(labels, logits)

  return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            #host_call=host_call,
            eval_metric_ops=eval_metrics)

def build_estimator(model_dir, config=None):
  #strategy = tf.contrib.distribute.MirroredStrategy()
  #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  #config.gpu_options.allow_growth = True
  config = tf.estimator.RunConfig(#train_distribute=strategy, eval_distribute=strategy,
    save_checkpoints_steps=100)
  #tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver()
  # config = tf.contrib.tpu.RunConfig(
  #   tpu_config=tpu_config.TPUConfig(num_shards=None, iterations_per_loop=100))
  #    eval_distribute=strategy)
  #return tf.estimator.Estimator(
  #    model_fn=_cnn_model_fn,
  #    model_dir=model_dir,
  #    config=config)
  return tf.contrib.tpu.TPUEstimator(
      model_fn=_cnn_model_fn,
      #model_dir=model_dir,
      use_tpu=True,
      config=config)
