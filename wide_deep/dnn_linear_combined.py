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
from tensorflow.python.training import training_util

# The default learning rates are a historical artifact of the initial
# implementation.
_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005
DNN_PAIRS_NUM = 18

def _linear_learning_rate(num_linear_feature_columns):
  default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
  return min(_LINEAR_LEARNING_RATE, default_learning_rate)

class CombinedOptimizer(tf.train.Optimizer):
  def __init__(self, linear_feature_columns=None):
    self.dnn_optimizer = optimizers.get_optimizer_instance('Adagrad', learning_rate=_DNN_LEARNING_RATE)
    # self.linear_optimizer = optimizers.get_optimizer_instance('Ftrl', learning_rate=_linear_learning_rate(len(linear_feature_columns)))
    self.linear_optimizer = optimizers.get_optimizer_instance('Ftrl', learning_rate = _LINEAR_LEARNING_RATE)
    self._name = 'combined'

  def compute_gradients(self, loss):
    pairs = self.dnn_optimizer.compute_gradients(loss, var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope='dnn'))
    linear_pairs = self.linear_optimizer.compute_gradients(loss, var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope='linear'))
    pairs.extend(linear_pairs)
    # for pair in linear_pairs:
    #   pairs.append(pair)
    return pairs
  
  def apply_gradients(self, grads_and_vars, global_step):
    dnn_pairs, linear_pairs = grads_and_vars[:DNN_PAIRS_NUM], grads_and_vars[DNN_PAIRS_NUM:]
    dnn_ops, linear_ops = self.dnn_optimizer.apply_gradients(dnn_pairs), self.linear_optimizer.apply_gradients(linear_pairs)

    train_ops = [dnn_ops, linear_ops]

    return control_flow_ops.group(*train_ops)

########################################################################

def _dnn_linear_combined_model_fn(
    features, labels, mode, head, num_workers, opt,
    linear_feature_columns=None,
    dnn_feature_columns=None, dnn_hidden_units=None,
    dnn_activation_fn=nn.relu, dnn_dropout=None,
    input_layer_partitioner=None, config=None):

  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = input_layer_partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  # combined_optimizer = CombinedOptimizer(linear_feature_columns)
  # sync_optimizer = tf.train.SyncReplicasOptimizer(combined_optimizer, replicas_to_aggregate=num_workers, total_num_replicas=num_workers)

  dnn_parent_scope = 'dnn'
  linear_parent_scope = 'linear'

  dnn_partitioner = (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas))
  with variable_scope.variable_scope(
      dnn_parent_scope,
      values=tuple(six.itervalues(features)),
      partitioner=dnn_partitioner):

    dnn_logit_fn = dnn._dnn_logit_fn_builder(
        units=head.logits_dimension,
        hidden_units=dnn_hidden_units,
        feature_columns=dnn_feature_columns,
        activation_fn=dnn_activation_fn,
        dropout=dnn_dropout,
        input_layer_partitioner=input_layer_partitioner)
    dnn_logits = dnn_logit_fn(features=features, mode=mode)

  with variable_scope.variable_scope(
      linear_parent_scope,
      values=tuple(six.itervalues(features)),
      partitioner=input_layer_partitioner) as scope:
    logit_fn = linear._linear_logit_fn_builder(
        units=head.logits_dimension,
        feature_columns=linear_feature_columns)
    linear_logits = logit_fn(features=features)

  logits = dnn_logits + linear_logits


  def _train_op_fn(loss):
    """Returns the op to optimize the loss."""
    global_step = training_util.get_global_step()

    pairs = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(pairs, global_step)

    # train_op = control_flow_ops.group(*train_ops)
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
               num_workers,
               model_dir=None,
               linear_feature_columns=None,
               dnn_feature_columns=None,
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

    self.optimizer = self.create_optimizer(num_workers, linear_feature_columns)

    def _model_fn(features, labels, mode, config):
      return _dnn_linear_combined_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          linear_feature_columns=linear_feature_columns,
          dnn_feature_columns=dnn_feature_columns,
          dnn_hidden_units=dnn_hidden_units,
          dnn_activation_fn=dnn_activation_fn,
          dnn_dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config,
          num_workers=num_workers,
          opt=self.optimizer)

    super(DNNLinearCombinedClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)

  def create_optimizer(self, num_workers, linear_feature_columns=None):
    opt = CombinedOptimizer(linear_feature_columns)
    return tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_workers, total_num_replicas=num_workers)

  def get_optimizer(self):
    return self.optimizer

