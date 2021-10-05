# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Supported models."""
import models.att_mh_rnn as att_mh_rnn
import models.att_rnn as att_rnn
import models.cnn as cnn
import models.crnn as crnn
import models.dnn as dnn
import models.dnn_raw as dnn_raw
import models.ds_cnn as ds_cnn
import models.ds_tc_resnet as ds_tc_resnet
import models.gru as gru
import models.inception as inception
import models.inception_resnet as inception_resnet
import models.lstm as lstm
import models.mobilenet as mobilenet
import models.mobilenet_v2 as mobilenet_v2
# import models.svdf as svdf
# import models.svdf_resnet as svdf_resnet
import models.tc_resnet as tc_resnet
import models.xception as xception

# dict with supported models
MODELS = {
    'att_mh_rnn': att_mh_rnn.model,
    'att_rnn': att_rnn.model,
    'dnn': dnn.model,
    'dnn_raw': dnn_raw.model,
    'ds_cnn': ds_cnn.model,
    'cnn': cnn.model,
    'tc_resnet': tc_resnet.model,
    'crnn': crnn.model,
    'gru': gru.model,
    'lstm': lstm.model,
    # 'svdf': svdf.model,
    # 'svdf_resnet': svdf_resnet.model,
    'mobilenet': mobilenet.model,
    'mobilenet_v2': mobilenet_v2.model,
    'xception': xception.model,
    'inception': inception.model,
    'inception_resnet': inception_resnet.model,
    'ds_tc_resnet': ds_tc_resnet.model,
}
