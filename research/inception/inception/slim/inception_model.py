# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inception-v3 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """Latest Inception from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for name_scope.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.name_scope(scope, 'inception_v3', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):
        # 299 x 299 x 3
        end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        # 149 x 149 x 32
        end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # 147 x 147 x 32
        end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2')
        print('AFTER CONV2 SHAPE: {}'.format(end_points['conv2'].get_shape()))
        # 147 x 147 x 64
        end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # 73 x 73 x 64
        end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3')
        # 73 x 73 x 80.
        end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4')
        # 71 x 71 x 192.
        end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # 35 x 35 x 192.
        net = end_points['pool2']
        print('NET SHAPE: {}'.format(net.get_shape()))
      # Inception blocks
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_35x35x256a'):
          print('SCOPE IS mixed_35x35x256a')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            print('BRANCH 5x5 SHAPE pt 1: {}'.format(branch5x5.get_shape()))
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
            print('BRANCH 5x5 SHAPE part 2: {}'.format(branch5x5.get_shape()))
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            print('BRANCH 3x3dbl SHAPE pt 1: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 2: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 3: {}'.format(branch3x3dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x256a'] = net
          print('BRANCH mixed_35x35x256a SHAPE: {}'.format(net.get_shape()))
        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288a'):
          print('SCOPE IS mixed_35x35x288a')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            print('BRANCH 5x5 SHAPE pt 1: {}'.format(branch5x5.get_shape()))
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
            print('BRANCH 5x5 SHAPE pt 2: {}'.format(branch5x5.get_shape()))
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            print('BRANCH 3x3dbl SHAPE pt 1: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 2: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 3: {}'.format(branch3x3dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288a'] = net
          print('BRANCH mixed_35x35x288a SHAPE: {}'.format(net.get_shape()))
        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288b'):
          print('SCOPE IS mixed_35x35x288b')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            print('BRANCH 5x5 SHAPE pt 1: {}'.format(branch5x5.get_shape()))
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
            print('BRANCH 5x5 SHAPE pt 2: {}'.format(branch5x5.get_shape()))
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            print('BRANCH 3x3dbl SHAPE pt 1: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 2: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 3: {}'.format(branch3x3dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288b'] = net
          print('BRANCH mixed_35x35x288b SHAPE: {}'.format(net.get_shape()))
        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768a'):
          print('SCOPE IS mixed_17x17x768a')
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
            print('BRANCH 3x3 SHAPE: {}'.format(branch3x3.get_shape()))
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            print('BRANCH 3x3dbl SHAPE pt 1: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 2: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID')
            print('BRANCH 3x3dbl SHAPE pt 3: {}'.format(branch3x3dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
            print('BRANCH POOL SHAPE: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_17x17x768a'] = net
          print('BRANCH mixed_17x17x768a SHAPE: {}'.format(net.get_shape()))
        # mixed4: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768b'):
          print('SCOPE IS mixed_17x17x768b')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 128, [1, 1])
            print('BRANCH 7x7 SHAPE part 1: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
            print('BRANCH 7x7 SHAPE part 2: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
            print('BRANCH 7x7 SHAPE part 3: {}'.format(branch7x7.get_shape()))
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 128, [1, 1])
            print('BRANCH 7x7dbl SHAPE part 1: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 2: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 3: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 4: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 5: {}'.format(branch7x7dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768b'] = net
          print('BRANCH mixed_17x17x768b SHAPE: {}'.format(net.get_shape()))
        # mixed_5: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768c'):
          print('SCOPE IS mixed_17x17x768c')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            print('BRANCH 7x7 SHAPE part 1: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            print('BRANCH 7x7 SHAPE part 2: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
            print('BRANCH 7x7 SHAPE part 3: {}'.format(branch7x7.get_shape()))
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            print('BRANCH 7x7dbl SHAPE part 1: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 2: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 3: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 4: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 5: {}'.format(branch7x7dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768c'] = net
          print('BRANCH mixed_17x17x768c SHAPE: {}'.format(net.get_shape()))
        # mixed_6: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768d'):
          print('SCOPE IS mixed_17x17x768d')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            print('BRANCH 7x7 SHAPE part 1: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            print('BRANCH 7x7 SHAPE part 2: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
            print('BRANCH 7x7 SHAPE part 3: {}'.format(branch7x7.get_shape()))
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            print('BRANCH 7x7dbl SHAPE part 1: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 2: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 3: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 4: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 5: {}'.format(branch7x7dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768d'] = net
          print('BRANCH mixed_17x17x768d SHAPE: {}'.format(net.get_shape()))
        # mixed_7: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768e'):
          print('SCOPE IS mixed_17x17x768e')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 7x7 SHAPE part 1: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 192, [1, 7])
            print('BRANCH 7x7 SHAPE part 2: {}'.format(branch7x7.get_shape()))
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
            print('BRANCH 7x7 SHAPE part 3: {}'.format(branch7x7.get_shape()))
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 7x7dbl SHAPE part 1: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 2: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 3: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            print('BRANCH 7x7dbl SHAPE part 4: {}'.format(branch7x7dbl.get_shape()))
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            print('BRANCH 7x7dbl SHAPE part 5: {}'.format(branch7x7dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768e'] = net
          print('BRANCH mixed_17x17x768e SHAPE: {}'.format(net.get_shape()))
        # Auxiliary Head logits
        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
        with tf.variable_scope('aux_logits'):
          aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                    padding='VALID')
          aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
          # Shape of feature map before the final layer.
          shape = aux_logits.get_shape()
          print('AUX_LOGITS SHAPE part 1: {}'.format(shape))
          aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01,
                                  padding='VALID')
          aux_logits = ops.flatten(aux_logits)
          print('AUX LOGITS shape part 2: {}'.format(aux_logits))
          aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                              stddev=0.001, restore=restore_logits)
          end_points['aux_logits'] = aux_logits
        # mixed_8: 8 x 8 x 1280.
        # Note that the scope below is not changed to not void previous
        # checkpoints.
        # (TODO) Fix the scope when appropriate.
        with tf.variable_scope('mixed_17x17x1280a'):
          print('SCOPE IS mixed_17x17x1280a')
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 3x3 SHAPE part 1: {}'.format(branch3x3.get_shape()))
            branch3x3 = ops.conv2d(branch3x3, 320, [3, 3], stride=2,
                                   padding='VALID')
            print('BRANCH 3x3 SHAPE part 1: {}'.format(branch3x3.get_shape()))
          with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = ops.conv2d(net, 192, [1, 1])
            print('BRANCH 7x3x3 SHAPE part 1: {}'.format(branch7x7x3.get_shape()))
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [1, 7])
            print('BRANCH 7x3x3 SHAPE part 2: {}'.format(branch7x7x3.get_shape()))
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [7, 1])
            print('BRANCH 7x3x3 SHAPE part 3: {}'.format(branch7x7x3.get_shape()))
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [3, 3],
                                     stride=2, padding='VALID')
            print('BRANCH 7x3x3 SHAPE part 4: {}'.format(branch7x7x3.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
            print('BRANCH POOL SHAPE: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
          end_points['mixed_17x17x1280a'] = net
          print('BRANCH mixed_17x17x1280a SHAPE: {}'.format(net.get_shape()))
        # mixed_9: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048a'):
          print('SCOPE IS mixed_8x8x2048a')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            print('BRANCH 3x3 SHAPE part 1: {}'.format(branch3x3.get_shape()))
            branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, 384, [1, 3]),
                                                  ops.conv2d(branch3x3, 384, [3, 1])])
            print('BRANCH 3x3 SHAPE part 2: {}'.format(branch3x3.get_shape()))
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            print('BRANCH 3x3dbl SHAPE pt 1: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 2: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = tf.concat(axis=3, values=[ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     ops.conv2d(branch3x3dbl, 384, [3, 1])])
            print('BRANCH 3x3dbl SHAPE pt 3: {}'.format(branch3x3dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048a'] = net
          print('BRANCH mixed_8x8x2048a SHAPE: {}'.format(net.get_shape()))
        # mixed_10: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048b'):
          print('SCOPE IS mixed_8x8x2048b')
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
            print('BRANCH 1x1 SHAPE: {}'.format(branch1x1.get_shape()))
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            print('BRANCH 3x3 SHAPE part 1: {}'.format(branch3x3.get_shape()))
            branch3x3 = tf.concat(axis=3, values=[ops.conv2d(branch3x3, 384, [1, 3]),
                                                  ops.conv2d(branch3x3, 384, [3, 1])])
            print('BRANCH 3x3 SHAPE part 2: {}'.format(branch3x3.get_shape()))
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            print('BRANCH 3x3dbl SHAPE pt 1: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            print('BRANCH 3x3dbl SHAPE pt 2: {}'.format(branch3x3dbl.get_shape()))
            branch3x3dbl = tf.concat(axis=3, values=[ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     ops.conv2d(branch3x3dbl, 384, [3, 1])])
            print('BRANCH 3x3dbl SHAPE pt 3: {}'.format(branch3x3dbl.get_shape()))
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            print('BRANCH POOL SHAPE part 1: {}'.format(branch_pool.get_shape()))
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
            print('BRANCH POOL SHAPE part 2: {}'.format(branch_pool.get_shape()))
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048b'] = net
          print('BRANCH mixed_8x8x2048b SHAPE: {}'.format(net.get_shape()))
        # Final pooling and prediction
        with tf.variable_scope('logits'):
          shape = net.get_shape()
          print('SHAPE BEFORE DROPOUT: {}'.format(shape))
          net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = ops.dropout(net, dropout_keep_prob, scope='dropout')
          net = ops.flatten(net, scope='flatten')
          # 2048
          logits = ops.fc(net, num_classes, activation=None, scope='logits',
                          restore=restore_logits)
          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits, end_points


def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope
