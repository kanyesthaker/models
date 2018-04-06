# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed training and evaluation of a wide and deep model."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Communication protocol to use in distributed """
                           """execution (default grpc) """)

tf.app.flags.DEFINE_integer('train_steps', 10000, 'Number of batches to run.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in SyncReplicasOptimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the SyncReplicasOptimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")

tf.app.flags.DEFINE_string('data_dir', 'wide_deep/data', """Data directory""")
tf.app.flags.DEFINE_string('model_dir', '/tmp/census_wide_and_deep_model', """directory for storing the model""")


# Define features for the model
def census_model_config():
  """Configuration for the census Wide & Deep model.

  Returns:
    columns: Column names to retrieve from the data source
    label_column: Name of the label column
    wide_columns: List of wide columns
    deep_columns: List of deep columns
    categorical_column_names: Names of the categorical columns
    continuous_column_names: Names of the continuous columns
  """
  # 1. Categorical base columns.
  gender = tf.contrib.layers.sparse_column_with_keys(
      column_name="gender", keys=["female", "male"])
  race = tf.contrib.layers.sparse_column_with_keys(
      column_name="race",
      keys=["Amer-Indian-Eskimo",
            "Asian-Pac-Islander",
            "Black",
            "Other",
            "White"])
  education = tf.contrib.layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000)
  marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(
      "marital_status", hash_bucket_size=100)
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100)
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100)
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000)
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000)

  # 2. Continuous base columns.
  age = tf.contrib.layers.real_valued_column("age")
  age_buckets = tf.contrib.layers.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  education_num = tf.contrib.layers.real_valued_column("education_num")
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

  wide_columns = [
      gender, native_country, education, occupation, workclass,
      marital_status, relationship, age_buckets,
      tf.contrib.layers.crossed_column([education, occupation],
                                       hash_bucket_size=int(1e4)),
      tf.contrib.layers.crossed_column([native_country, occupation],
                                       hash_bucket_size=int(1e4)),
      tf.contrib.layers.crossed_column([age_buckets, race, occupation],
                                       hash_bucket_size=int(1e6))]

  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(marital_status, dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(race, dimension=8),
      tf.contrib.layers.embedding_column(native_country, dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age, education_num, capital_gain, capital_loss, hours_per_week]

  # Define the column names for the data sets.
  columns = ["age", "workclass", "fnlwgt", "education", "education_num",
             "marital_status", "occupation", "relationship", "race", "gender",
             "capital_gain", "capital_loss", "hours_per_week",
             "native_country", "income_bracket"]
  label_column = "label"
  categorical_columns = ["workclass", "education", "marital_status",
                         "occupation", "relationship", "race", "gender",
                         "native_country"]
  continuous_columns = ["age", "education_num", "capital_gain",
                        "capital_loss", "hours_per_week"]

  return (columns, label_column, wide_columns, deep_columns,
          categorical_columns, continuous_columns)


class CensusDataSource(object):
  """Source of census data."""

  def __init__(self, data_dir,
               columns, label_column,
               categorical_columns, continuous_columns):

    train_file_path = os.path.join(data_dir, "adult.data")
    train_file = open(train_file_path)

    test_file_path = os.path.join(data_dir, "adult.test")
    test_file = open(test_file_path)

    import pandas  # pylint: disable=g-import-not-at-top
    self._df_train = pandas.read_csv(train_file, names=columns,
                                     skipinitialspace=True)
    self._df_test = pandas.read_csv(test_file, names=columns,
                                    skipinitialspace=True, skiprows=1)

    # Remove the NaN values in the last rows of the tables
    self._df_train = self._df_train[:-1]
    self._df_test = self._df_test[:-1]

    # Apply the threshold to get the labels.
    income_thresh = lambda x: ">50K" in x
    self._df_train[label_column] = (
        self._df_train["income_bracket"].apply(income_thresh)).astype(int)
    self._df_test[label_column] = (
        self._df_test["income_bracket"].apply(income_thresh)).astype(int)

    self.label_column = label_column
    self.categorical_columns = categorical_columns
    self.continuous_columns = continuous_columns

  def input_train_fn(self):
    return self._input_fn(self._df_train)

  def input_test_fn(self):
    return self._input_fn(self._df_test)

  # TODO(cais): Turn into minibatch feeder
  def _input_fn(self, df):
    """Input data function.

    Creates a dictionary mapping from each continuous feature column name
    (k) to the values of that column stored in a constant Tensor.

    Args:
      df: data feed

    Returns:
      feature columns and labels
    """
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in self.continuous_columns}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in self.categorical_columns}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[self.label_column].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def run(cluster_spec):
  """Train WD on a dataset for a number of steps."""

  os.environ["TF_CONFIG"] = json.dumps({
    "cluster": cluster_spec.as_dict(),
    "task": {
        "index": FLAGS.task_id,
        "type": FLAGS.job_name
    }
  })

  (columns, label_column, wide_columns, deep_columns, categorical_columns,
   continuous_columns) = census_model_config()

  census_data_source = CensusDataSource(FLAGS.data_dir,
                                        columns, label_column,
                                        categorical_columns,
                                        continuous_columns)

  config = tf.estimator.RunConfig(save_checkpoints_steps=None, save_checkpoints_secs=None)

  estimator = tf.estimator.DNNLinearCombinedClassifier(
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=[100, 75, 50, 25],
      config=config)

  timeline_hook = tf.train.ProfilerHook(save_steps=1, show_dataflow=True, show_memory=False)

  train_spec = tf.estimator.TrainSpec(input_fn=census_data_source.input_train_fn, max_steps=FLAGS.train_steps, hooks=[timeline_hook])
  eval_spec = tf.estimator.EvalSpec(input_fn=census_data_source.input_test_fn)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
