# Copyright 2016 Google Inc. All Rights Reserved.
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
# pylint: disable=line-too-long
"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import wd_distributed_train

FLAGS = tf.app.flags.FLAGS

def main(unused_args):
  assert FLAGS.job_name in ['ps', 'worker', 'chief'], 'job_name must be ps or worker'

  # Extract all the hostnames for the ps and worker jobs to construct the
  # cluster spec.
  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  tf.logging.info('PS hosts are: %s' % ps_hosts)
  tf.logging.info('Worker hosts are: %s' % worker_hosts[1:])
  tf.logging.info('Chief host is: %s' % worker_hosts[0])

  chief_worker = worker_hosts[0]
  worker_hosts = worker_hosts[1:]

  if len(worker_hosts) > 0:
    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts, 'chief': [chief_worker]})
  else:
    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'chief': [chief_worker]})
    
  wd_distributed_train.run(cluster_spec, len(worker_hosts) + 1)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
