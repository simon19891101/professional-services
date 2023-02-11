# Copyright 2021 Google LLC. All Rights Reserved.
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

"""
Model training program.
"""

from typing import Dict, Tuple, Optional, List, Iterable

import argparse
import datetime
import json
import logging
import os

import yaml
import tensorflow as tf
import numpy as np
import lightgbm as lgb
import pandas as pd
from google.cloud import aiplatform_v1beta1 as aip
from sklearn import metrics as sk_metrics
from sklearn import model_selection

import hypertune

# pylint: disable=logging-fstring-interpolation


################################################################################
# Model serialization code
# ###############################################################################

MODEL_FILENAME = 'model.txt'
FEATURE_IMPORTANCE_FILENAME = 'feature_importance.csv'
INSTANCE_SCHEMA_FILENAME = 'instance_schema.yaml'
PROB_THRESHOLD = 0.5


def _save_lgb_model(model: lgb.Booster, model_store: str):
  """Export trained lgb model."""
  file_path = os.path.join(model_store, MODEL_FILENAME)
  model.save_model(MODEL_FILENAME)
  tf.io.gfile.copy(MODEL_FILENAME, file_path, overwrite=True)


def _save_lgb_feature_importance(model: lgb.Booster, model_store: str):
  """Export feature importance info of trained lgb model."""
  file_path = os.path.join(model_store, FEATURE_IMPORTANCE_FILENAME)
  # Pandas can save to GCS directly
  pd.DataFrame(
      {
          'feature': model.feature_name(),
          'importance': model.feature_importance()
      }
  ).to_csv(file_path, index=False)


def _save_metrics(metrics: dict, output_path: str):
  """Export the metrics of trained lgb model."""
  with tf.io.gfile.GFile(output_path, 'w') as eval_file:
    eval_file.write(json.dumps(metrics))


def _save_analysis_schema(df: pd.DataFrame, model_store: str):
  """Export instance schema for model monitoring service."""
  file_path = os.path.join(model_store, INSTANCE_SCHEMA_FILENAME)
  # create feature schema
  properties = {}
  types_info = df.dtypes

  for i in range(len(types_info)):
    if types_info.values[i] == object:
      properties[types_info.index[i]] = {'type': 'string', 'nullable': True}
    else:
      properties[types_info.index[i]] = {'type': 'number', 'nullable': True}

  spec = {
      'type': 'object',
      'properties': properties,
      'required': df.columns.tolist()
  }

  with tf.io.gfile.GFile(file_path, 'w') as file:
    yaml.dump(spec, file)


################################################################################
# Data loading
# ###############################################################################

def _split_features_label_columns(df: pd.DataFrame,
                                  target_label: str
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Split dataset into features and target."""
  y = df[target_label]
  x = df.drop(target_label, axis=1)

  return x, y


def load_csv_dataset(data_uri_pattern: str,
                     target_label: str,
                     features: List[str],
                     data_schema: Optional[str] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Load CSV data into features and label DataFrame."""
  all_files = tf.io.gfile.glob(data_uri_pattern)

  if data_schema:
    # [[name, dtype],]
    fields = dict(field.split(':') for field in data_schema.split(';'))
    field_names = fields.keys()
    df = pd.concat((pd.read_csv('gs://' + f, names=field_names, dtype=fields)
                    for f in all_files), ignore_index=True)
  else:
    df = pd.concat((pd.read_csv('gs://' + f)
                    for f in all_files), ignore_index=True)
  # Shuffle
  df = df.sample(frac=1).reset_index(drop=True)

  logging.info(df.head(2))

  x, y = _split_features_label_columns(df, target_label)
  if features:
    x = x[features.split(',')]

  return x, y


################################################################################
# Model training
# ###############################################################################


def _evaluate_binary_classification(model: lgb.Booster,
                                    x: pd.DataFrame,
                                    y: pd.DataFrame) -> Dict[str, object]:
  """Perform evaluation of binary classification model."""
  # get roc curve metrics, down sample to avoid hitting MLMD 64k size limit
  roc_size = int(x.shape[0] * 1 / 3)
  y_hat = model.predict(x)
  pred = (y_hat > PROB_THRESHOLD).astype(int)

  fpr, tpr, thresholds = sk_metrics.roc_curve(
      y_true=y[:roc_size], y_score=y_hat[:roc_size], pos_label=True)

  # get classification metrics
  au_roc = sk_metrics.roc_auc_score(y, y_hat)
  au_prc = sk_metrics.average_precision_score(y, y_hat)
  classification_metrics = sk_metrics.classification_report(
      y, pred, output_dict=True)
  confusion_matrix = sk_metrics.confusion_matrix(y, pred, labels=[0, 1])

  metrics = {
      'classification_report': classification_metrics,
      'confusion_matrix': confusion_matrix.tolist(),
      'au_roc': au_roc,
      'au_prc': au_prc,
      'fpr': fpr.tolist(),
      'tpr': tpr.tolist(),
      'thresholds': thresholds.tolist()
  }

  logging.info(f'The evaluation report: {metrics}')

  return metrics


def lgb_training(lgb_train: lgb.Dataset,
                 lgb_val: lgb.Dataset,
                 num_boost_round: int,
                 num_leaves: int,
                 max_depth: int,
                 min_data_in_leaf: int) -> lgb.Booster:
  """Train lgb model given datasets and parameters."""
  # train the model
  params = {
      'objective': 'binary',
      'is_unbalance': True,
      'boosting_type': 'gbdt',
      'metric': ['auc'],
      'num_leaves': num_leaves,
      'max_depth': max_depth,
      'min_data_in_leaf': min_data_in_leaf
  }

  eval_results = {}  # to record eval results
  model = lgb.train(params=params,
                    num_boost_round=num_boost_round,
                    train_set=lgb_train,
                    valid_sets=[lgb_val, lgb_train],
                    valid_names=['test', 'train'],
                    evals_result=eval_results,
                    verbose_eval=True)

  return model


################################################################################
# Main Logic.
# ###############################################################################

def train(args: argparse.Namespace):
  """The main training logic."""

  if 'AIP_MODEL_DIR' not in os.environ:
    raise KeyError(
        'The `AIP_MODEL_DIR` environment variable has not been set. '
        'See https://cloud.google.com/ai-platform-unified/docs/tutorials/'
        'image-recognition-custom/training'
    )
  output_model_directory = os.environ['AIP_MODEL_DIR']

  logging.info(f'AIP_MODEL_DIR: {output_model_directory}')
  logging.info(f'training_data_uri: {args.training_data_uri}')

  # prepare the data
  x_train, y_train = load_csv_dataset(
      data_uri_pattern=args.training_data_uri,
      data_schema=args.training_data_schema,
      target_label=args.label,
      features=args.features)

  # validation data
  x_train, x_val, y_train, y_val = model_selection.train_test_split(
      x_train,
      y_train,
      test_size=0.2,
      random_state=np.random.RandomState(42))

  # test data
  x_val, x_test, y_val, y_test = model_selection.train_test_split(
      x_val,
      y_val,
      test_size=0.5,
      random_state=np.random.RandomState(42))

  lgb_train = lgb.Dataset(x_train, y_train, categorical_feature='auto')
  lgb_val = lgb.Dataset(x_val, y_val, categorical_feature='auto')
  
  best_param_values = {
      'num_leaves': int(args.num_leaves_hp_param_min +
                     args.num_leaves_hp_param_max) // 2,
      'max_depth': int(args.max_depth_hp_param_min +
                    args.max_depth_hp_param_max) // 2
  }

  model = lgb_training(
      lgb_train=lgb_train,
      lgb_val=lgb_val,
      num_boost_round=int(args.num_boost_round),
      min_data_in_leaf=int(args.min_data_in_leaf),
      **best_param_values)

  # save eval metrics
  metrics = _evaluate_binary_classification(model, x_test, y_test)

  # DEFINE METRIC
  hp_metric = metrics['au_roc']

  hpt = hypertune.HyperTune()

  hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='au_roc',
    metric_value=hp_metric
  )

  metric_files = str(os.listdir(os.path.dirname(hpt.metric_path)))
  logging.info(f"metric_files: {metric_files}")



if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser()
  # For training data
  parser.add_argument('--training_data_uri', type=str,
                      help='The training dataset location in GCS.')
  parser.add_argument('--training_data_schema', type=str, default='',
                      help='The schema of the training dataset. The'
                           'example schema: name:type;')
  parser.add_argument('--features', type=str, default='',
                      help='The column names of features to be used.')
  parser.add_argument('--label', type=str, default='',
                      help='The column name of label in the dataset.')

  # For model hyperparameter
  parser.add_argument('--min_data_in_leaf', default=5, type=float,
                      help='Minimum number of observations that must '
                           'fall into a tree node for it to be added.')
  parser.add_argument('--num_boost_round', default=300, type=float,
                      help='Number of boosting iterations.')
  parser.add_argument('--max_depth_hp_param_min', default=-1, type=float,
                      help='Max tree depth for base learners, <=0 means no '
                           'limit. Min value for hyperparam param')
  parser.add_argument('--max_depth_hp_param_max', default=4, type=float,
                      help='Max tree depth for base learners, <=0 means no '
                           'limit.  Max value for hyperparam param')
  parser.add_argument('--num_leaves_hp_param_min', default=6, type=float,
                      help='Maximum tree leaves for base learners. '
                           'Min value for hyperparam param.')
  parser.add_argument('--num_leaves_hp_param_max', default=10, type=float,
                      help='Maximum tree leaves for base learners. '
                           'Max value for hyperparam param.')

  logging.info(parser.parse_args())
  known_args, _ = parser.parse_known_args()
  train(known_args)
