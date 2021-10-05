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

"""Xception - reduced version of keras/applications/xception.py."""
from re import T
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.recurrent import LSTM,GRU
from layers import modes
from layers import speech_features
from layers.compat import tf
import models.model_utils as utils
from tensorflow.keras import regularizers
from keras.layers import GRU
from models.attention import Attention, MyMultiHeadAttention, MySelfAttention, MySelfAttention2
from models.custom_attention import attention
import kapre
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Dense
from spec_augment import SpecAugment
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from keras_multi_head import MultiHeadAttention
from keras_self_attention import ScaledDotProductAttention


def model_parameters(parser_nn):
  """Xception model parameters.

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """
  parser_nn.add_grument(
      '--dt_type',
      type=str,
      defalut='mel',
      help='mel,mfcc'
  )
  parser_nn.add_grument(
      '--att_type',
      type=str,
      default='None',
      help='self_att,multi_head,att'
  )
  parser_nn.add_argument(
      '--cnn1_kernel_sizes',
      type=str,
      default='5',
      help='Kernel_size of the conv block 1',
  )
  parser_nn.add_argument(
      '--cnn1_filters',
      type=str,
      default='32',
      help='Number of filters in the conv block 1',
  )
  parser_nn.add_argument(
      '--stride1',
      type=int,
      default=2,
      help='Stride of pooling layer after conv block 1',
  )
  parser_nn.add_argument(
      '--stride2',
      type=int,
      default=2,
      help='Stride of pooling layer after conv block 2 xception',
  )
  parser_nn.add_argument(
      '--stride3',
      type=int,
      default=2,
      help='Stride of pooling layer after conv block 3 xception',
  )
  parser_nn.add_argument(
      '--stride4',
      type=int,
      default=2,
      help='Stride of pooling layer after conv block 4 xception',
  )
  parser_nn.add_argument(
      '--cnn2_kernel_sizes',
      type=str,
      default='5',
      help='Kernel_size of the conv block 2 xception',
  )
  parser_nn.add_argument(
      '--cnn2_filters',
      type=str,
      default='32',
      help='Number of filters in the conv block 2 xception',
  )
  parser_nn.add_argument(
      '--cnn3_kernel_sizes',
      type=str,
      default='5',
      help='Kernel size of the conv block 3 xception',
  )
  parser_nn.add_argument(
      '--cnn3_filters',
      type=str,
      default='32',
      help='Number of filters in the third conv block 3 xception',
  )
  parser_nn.add_argument(
      '--cnn4_kernel_sizes',
      type=str,
      default='5',
      help='Kernel sizes of the conv block 4 xception',
  )
  parser_nn.add_argument(
      '--cnn4_filters',
      type=str,
      default='32',
      help='Number of filters in the conv block4 xception',
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.0,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--bn_scale',
      type=int,
      default=1,
      help='If True, multiply by gamma. If False, gamma is not used. '
      'When the next layer is linear (also e.g. nn.relu), this can be disabled'
      'since the scaling will be done by the next layer.',
  )
  parser_nn.add_argument(
      '--units2',
      type=str,
      default='64',
      help='Number of units in the last set of hidden layers',
  )


def block(net, kernel_sizes, filters, dropout, bn_scale=False):
  """Utility function to apply conv + BN.

  Arguments:
    net: input tensor.
    kernel_sizes: size of convolution kernel.
    filters: filters in `Conv2D`.
    dropout: percentage of dropped data
    bn_scale: scale batch normalization.

  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  if not filters:
    return net

  net_residual = net
  # project
  net_residual = tf.keras.layers.Conv2D(
      filters[-1],
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None)(
          net_residual)
  net_residual = tf.keras.layers.BatchNormalization(scale=bn_scale)(
      net_residual)

  for i, (kernel_size, filters) in enumerate(zip(kernel_sizes, filters)):

    net = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(kernel_size, 1),
        activation=None,
        use_bias=False,
        padding='same')(net)

    net = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None)(net)
    net = tf.keras.layers.BatchNormalization(scale=bn_scale)(net)

    # in the bottom of this function we add residual connection
    # and then apply activation with dropout
    # so no need to do another activation and dropout in the end of this loop
    if i != len(kernel_sizes)-1:
      net = tf.keras.layers.Activation('relu')(net)
      net = tf.keras.layers.Dropout(dropout)(net)

  net = tf.keras.layers.Add()([net_residual, net])
  net = tf.keras.layers.Activation('relu')(net)
  net = tf.keras.layers.Dropout(dropout)(net)
  return net


def model(flags):
  """Xception model.

  It is based on papers:
  Xception: Deep Learning with Depthwise Separable Convolutions
      https://arxiv.org/abs/1610.02357
  MatchboxNet: 1D Time-Channel Separable Convolutional
  Neural Network Architecture for Speech Commands Recognition
  https://arxiv.org/pdf/2004.08531
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
  # input_audio = tf.keras.layers.Input(
  #     shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
  #     batch_size=flags.batch_size)
  # net = input_audio

  # if flags.preprocess == 'raw':
  #   # it is a self contained model, user need to feed raw audio only
  #   net = speech_features.SpeechFeatures(
  #       speech_features.SpeechFeatures.get_params(flags))(
  #           net)
  input_audio = tf.keras.layers.Input(shape=(16000,))
  net = input_audio
  if flags.dt_type=='mfcc':
      if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
        net = speech_features.SpeechFeatures(
            speech_features.SpeechFeatures.get_params(flags))(net)
  elif flags.dt_type=='mel':
    input_audio=(1,16000)
    i=get_melspectrogram_layer(input_shape=input_audio,
                                n_mels=80,
                                pad_end=True,
                                n_fft=1024,
                                hop_length=128,
                                sample_rate=16000,
                                mel_f_max=8000,
                                mel_f_min=40.0,
                                return_decibel=True,
                                input_data_format='channels_first',
                                output_data_format='channels_last')
    # net=LayerNormalization(axis=2,name='batch_norm')(i.output)
    # net=tf.keras.backend.permute_dimensions(i.output,[0,1,3,2])
    net = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim1')(i.output)
    net=SpecAugment()(net,training=True)
    net=L.Lambda(lambda q:K.expand_dims(q,axis=2),name='expand_midle_dim')(net)

  # [batch, time, feature]
  if flags.dt_type=='mfcc': 
    net = tf.keras.backend.expand_dims(net, axis=2)
  # [batch, time, 1, feature]

  # conv block
  for kernel_size, filters in zip(
      utils.parse(flags.cnn1_kernel_sizes), utils.parse(flags.cnn1_filters)):
    net = tf.keras.layers.Conv2D(
        filters, (kernel_size, 1),
        use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.Activation('relu')(net)
    # [batch, time, 1, feature]

  if flags.stride1 > 1:
    net = tf.keras.layers.MaxPooling2D((3, 1),
                                       strides=(flags.stride1, 1),
                                       padding='valid')(
                                           net)

  net = block(net, utils.parse(flags.cnn2_kernel_sizes),
              utils.parse(flags.cnn2_filters), flags.dropout, flags.bn_scale)
  if flags.stride2 > 1:
    net = tf.keras.layers.MaxPooling2D((3, 1),
                                       strides=(flags.stride2, 1),
                                       padding='valid')(
                                           net)

  net = block(net, utils.parse(flags.cnn3_kernel_sizes),
              utils.parse(flags.cnn3_filters), flags.dropout, flags.bn_scale)
  if flags.stride3 > 1:
    net = tf.keras.layers.MaxPooling2D((3, 1),
                                       strides=(flags.stride3, 1),
                                       padding='valid')(
                                           net)

  net = block(net, utils.parse(flags.cnn4_kernel_sizes),
              utils.parse(flags.cnn4_filters), flags.dropout, flags.bn_scale)
  if flags.stride4 > 1:
    net = tf.keras.layers.MaxPooling2D((3, 1),
                                       strides=(flags.stride4, 1),
                                       padding='valid')(
                                           net)
  if flags.att_type!= 'None':     
    shape = net.shape
    net = tf.keras.layers.Reshape(
        (-1, shape[2]*shape[3]), name='reshape2')(net)
    # value=net
    gru = L.Bidirectional(
        GRU(16, return_sequences=True,recurrent_dropout=0.4,dropout=0.3,kernel_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay)), name="bi_gru_0")(net)
    (gru, forward_h, backward_h) = L.Bidirectional(
        GRU(16, return_sequences=True, return_state=True,recurrent_dropout=0.4,dropout=0.3,kernel_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay)), name="bi_gru_1")(gru)
    
    # state_h = L.Concatenate()([forward_h, backward_h])
    
    state_h = tf.concat([forward_h,backward_h],1)
    # state_c = L.Concatenate()([forward_c, backward_c])
    k_v = gru
    q_v = state_h
        # q_v=tf.repeat(q_v,repeats=[k_v.shape[1]],axis=0)
        # q_v=tf.reshape(q_v,shape=(-1,k_v.shape[1],state_h.shape[1]))

    # q_v = tf.reshape(q_v, shape=(-1, 1, q_v.shape[1]))
    q_v=tf.expand_dims(q_v, 1)
    if flags.att_type=='self_att':
      #   net=MySelfAttention(output_dim=net.shape[2])(net)
      net = ScaledDotProductAttention()([q_v, k_v, k_v])
      net=tf.reduce_sum(net,axis=1)
    net=tf.keras.layers.Dropout(flags.dropout)(net)
    net=Flatten()(net)
  else:
      net = tf.keras.layers.GlobalAveragePooling2D()(net)
      # [batch, filters]
      net = tf.keras.layers.Dropout(flags.dropout)(net)

  for units in utils.parse(flags.units2):
    net = tf.keras.layers.Dense(
        units=units, activation=None, use_bias=False)(
            net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.Activation('relu')(net)

  net = tf.keras.layers.Dense(flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  # [batch, label_count]
  if flags.dt_type=='mel':
    return tf.keras.Model(i.input, net)
  else:
    return tf.keras.Model(input_audio, net)
