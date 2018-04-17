# # Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

"""TensorFlow estimators for Linear and DNN joined training models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import six

import tensorflow as tf
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util

# The default learning rates are a historical artifact of the initial
# implementation.
_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005


class CombinedOptimizer(tf.train.Optimizer):
  def __init__(self):
    print('yo')

def _check_no_sync_replicas_optimizer(optimizer):
  if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
    raise ValueError(
        'SyncReplicasOptimizer does not support multi optimizers case. '
        'Therefore, it is not supported in DNNLinearCombined model. '
        'If you want to use this optimizer, please use either DNN or Linear '
        'model.')


def _linear_learning_rate(num_linear_feature_columns):
  """Returns the default learning rate of the linear model.

  The calculation is a historical artifact of this initial implementation, but
  has proven a reasonable choice.

  Args:
    num_linear_feature_columns: The number of feature columns of the linear
      model.

  Returns:
    A float.
  """
  default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
  return min(_LINEAR_LEARNING_RATE, default_learning_rate)


def _add_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)


def _dnn_linear_combined_model_fn(
    features, labels, mode, head,
    linear_feature_columns=None, linear_optimizer='Ftrl',
    dnn_feature_columns=None, dnn_optimizer='Adagrad', dnn_hidden_units=None,
    dnn_activation_fn=nn.relu, dnn_dropout=None,
    input_layer_partitioner=None, config=None):

  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))
  if not linear_feature_columns and not dnn_feature_columns:
    raise ValueError(
        'Either linear_feature_columns or dnn_feature_columns must be defined.')
  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = input_layer_partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  # Build DNN Logits.
  dnn_parent_scope = 'dnn'

  if not dnn_feature_columns:
    dnn_logits = None
  else:
    dnn_optimizer = optimizers.get_optimizer_instance(
        dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
    _check_no_sync_replicas_optimizer(dnn_optimizer)
    if not dnn_hidden_units:
      raise ValueError(
          'dnn_hidden_units must be defined when dnn_feature_columns is '
          'specified.')
    dnn_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas))
    with variable_scope.variable_scope(
        dnn_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=dnn_partitioner):

      dnn_logit_fn = dnn._dnn_logit_fn_builder(  # pylint: disable=protected-access
          units=head.logits_dimension,
          hidden_units=dnn_hidden_units,
          feature_columns=dnn_feature_columns,
          activation_fn=dnn_activation_fn,
          dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner)
      dnn_logits = dnn_logit_fn(features=features, mode=mode)

  linear_parent_scope = 'linear'

  if not linear_feature_columns:
    linear_logits = None
  else:
    linear_optimizer = optimizers.get_optimizer_instance(
        linear_optimizer,
        learning_rate=_linear_learning_rate(len(linear_feature_columns)))
    _check_no_sync_replicas_optimizer(linear_optimizer)
    with variable_scope.variable_scope(
        linear_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner) as scope:
      logit_fn = linear._linear_logit_fn_builder(  # pylint: disable=protected-access
          units=head.logits_dimension,
          feature_columns=linear_feature_columns)
      linear_logits = logit_fn(features=features)
      _add_layer_summary(linear_logits, scope.name)

  # Combine logits and build full model.
  if dnn_logits is not None and linear_logits is not None:
    logits = dnn_logits + linear_logits
  elif dnn_logits is not None:
    logits = dnn_logits
  else:
    logits = linear_logits

  def _train_op_fn(loss):
    """Returns the op to optimize the loss."""
    train_ops = []
    global_step = training_util.get_global_step()
    if dnn_logits is not None:
      train_ops.append(
          dnn_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=dnn_parent_scope)))
    if linear_logits is not None:
      train_ops.append(
          linear_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=linear_parent_scope)))

    train_op = control_flow_ops.group(*train_ops)
    with ops.control_dependencies([train_op]):
      with ops.colocate_with(global_step):
        return state_ops.assign_add(global_step, 1)

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)


class DNNLinearCombinedClassifier(estimator.Estimator):
  def __init__(self,
               model_dir=None,
               linear_feature_columns=None,
               linear_optimizer='Ftrl',
               dnn_feature_columns=None,
               dnn_optimizer='Adagrad',
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               input_layer_partitioner=None,
               config=None):

    linear_feature_columns = linear_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    self._feature_columns = (
        list(linear_feature_columns) + list(dnn_feature_columns))
    if not self._feature_columns:
      raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                       'must be defined.')
    if n_classes == 2:
      head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
          weight_column=weight_column,
          label_vocabulary=label_vocabulary)
    else:
      head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
          n_classes,
          weight_column=weight_column,
          label_vocabulary=label_vocabulary)
    def _model_fn(features, labels, mode, config):
      return _dnn_linear_combined_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          linear_feature_columns=linear_feature_columns,
          linear_optimizer=linear_optimizer,
          dnn_feature_columns=dnn_feature_columns,
          dnn_optimizer=dnn_optimizer,
          dnn_hidden_units=dnn_hidden_units,
          dnn_activation_fn=dnn_activation_fn,
          dnn_dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config)

    super(DNNLinearCombinedClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)

