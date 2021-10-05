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

"""Model based on combination of n by 1 convolutions with residual blocks."""

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
import numpy as np

def model_parameters(parser_nn):
  """Temporal Convolution Resnet model parameters.

  In more details parameters are described at:
  https://arxiv.org/pdf/1904.03814.pdf
  We converted model to Keras and made it compatible with TF V2
  https://github.com/hyperconnect/TC-ResNet


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
      '--channels',
      type=str,
      default='24, 36, 36, 48, 48, 72, 72',
      help='Number of channels per convolutional block (including first conv)',
  )
  parser_nn.add_argument(
      '--debug_2d',
      type=int,
      default=0,
      help='If 0 conv_kernel will be [3, 3], else conv_kernel [3, 1]',
  )
  parser_nn.add_argument(
      '--pool_size',
      type=str,
      default='',
      help="Pool size for example '4,4'",
  )
  parser_nn.add_argument(
      '--kernel_size',
      type=str,
      default='(9,1)',
      help='Kernel size of conv layer',
  )
  parser_nn.add_argument(
      '--pool_stride',
      type=int,
      default=0,
      help='Pool stride, for example 4',
  )
  parser_nn.add_argument(
      '--bn_momentum',
      type=float,
      default=0.997,
      help='Momentum for the moving average',
  )
  parser_nn.add_argument(
      '--bn_center',
      type=int,
      default=1,
      help='If True, add offset of beta to normalized tensor.'
      'If False, beta is ignored',
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
      '--bn_renorm',
      type=int,
      default=0,
      help='Whether to use Batch Renormalization',
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.2,
      help='Percentage of data dropped',
  )


def model(flags):
  """Temporal Convolution ResNet model.

  It is based on paper:
  Temporal Convolution for Real-time Keyword Spotting on Mobile Devices
  https://arxiv.org/pdf/1904.03814.pdf
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
#   input_audio = tf.keras.layers.Input(
#       shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
#       batch_size=flags.batch_size)
#   net = input_audio
  input_audio = tf.keras.layers.Input(shape=(16000,))
  net = input_audio
  if flags.dt_type=='mfcc':
      if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
        net = speech_features.SpeechFeatures(
            speech_features.SpeechFeatures.get_params(flags))(
                net)
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

  

  time_size, feature_size = net.shape[1],net.shape[-1]

  channels = utils.parse(flags.channels)
  if flags.dt_type=='mfcc':
    net = tf.keras.backend.expand_dims(net)

  if flags.debug_2d:
    conv_kernel = first_conv_kernel = (3, 3)
  else:
    net = tf.reshape(
        net, [-1, time_size, 1, feature_size])  # [batch, time, 1, feature]
    first_conv_kernel = (3, 1)
    conv_kernel = utils.parse(flags.kernel_size)

  net = tf.keras.layers.Conv2D(
      filters=channels[0],
      kernel_size=first_conv_kernel,
      strides=1,
      padding='same',
      activation='linear')(
          net)
  net = tf.keras.layers.BatchNormalization(
      momentum=flags.bn_momentum,
      center=flags.bn_center,
      scale=flags.bn_scale,
      renorm=flags.bn_renorm)(
          net)
  net = tf.keras.layers.Activation('relu')(net)

  if utils.parse(flags.pool_size):
    net = tf.keras.layers.AveragePooling2D(
        pool_size=utils.parse(flags.pool_size), strides=flags.pool_stride)(
            net)

  channels = channels[1:]

  # residual blocks
  for n in channels:
    if n != net.shape[-1]:
      stride = 2
      layer_in = tf.keras.layers.Conv2D(
          filters=n,
          kernel_size=1,
          strides=stride,
          padding='same',
          activation='linear')(
              net)
      layer_in = tf.keras.layers.BatchNormalization(
          momentum=flags.bn_momentum,
          center=flags.bn_center,
          scale=flags.bn_scale,
          renorm=flags.bn_renorm)(
              layer_in)
      layer_in = tf.keras.layers.Activation('relu')(layer_in)
    else:
      layer_in = net
      stride = 1

    net = tf.keras.layers.Conv2D(
        filters=n,
        kernel_size=conv_kernel,
        strides=stride,
        padding='same',
        activation='linear')(
            net)
    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(
            net)
    net = tf.keras.layers.Activation('relu')(net)

    net = tf.keras.layers.Conv2D(
        filters=n,
        kernel_size=conv_kernel,
        strides=1,
        padding='same',
        activation='linear')(
            net)
    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(
            net)

    # residual connection
    net = tf.keras.layers.Add()([net, layer_in])
    net = tf.keras.layers.Activation('relu')(net)

    if flags.att_type=='None':
        net = tf.keras.layers.AveragePooling2D(
        pool_size=net.shape[1:3], strides=1)(
          net)
        net = tf.keras.layers.Dropout(rate=flags.dropout)(net)
    # fully connected layer
        net = tf.keras.layers.Conv2D(
            filters=flags.label_count,
            # kernel_regularizer=regularizers.L2(0.001),
            kernel_size=1,
            strides=1,
            padding='same',
            activation='linear')(
                net)
        net = tf.reshape(net, shape=(-1, net.shape[3]))
  else:
       shape=net.shape
       net=tf.keras.layers.Reshape((-1,shape[2]*shape[3]),name='reshape')(net)
       gru = L.Bidirectional(
        GRU(16, return_sequences=True,recurrent_dropout=0.4,dropout=0.3,kernel_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay)), name="bi_gru_0")(net)
       (gru, forward_h, backward_h) = L.Bidirectional(
        GRU(16, return_sequences=True, return_state=True,recurrent_dropout=0.4,dropout=0.3,kernel_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay)), name="bi_gru_1")(gru)
       state_h = L.Concatenate()([forward_h, backward_h])
    # state_c = L.Concatenate()([forward_c, backward_c])
       k_v = gru
       q_v = state_h
        # q_v=tf.repeat(q_v,repeats=[k_v.shape[1]],axis=0)
        # q_v=tf.reshape(q_v,shape=(-1,k_v.shape[1],state_h.shape[1]))

       q_v = tf.reshape(q_v, shape=(-1, 1, q_v.shape[1]))
       if flags.att_type=='self_att':
            net = ScaledDotProductAttention()([q_v, k_v, k_v])
       elif flags.att_type=='multi_head':
            net=MyMultiHeadAttention(output_dim=125,num_head=5)(net)
       elif flags.att_type=='att':
            net=Attention(units=net.shape[1])(net)

       net = tf.keras.layers.Dropout(rate=flags.dropout)(net)
       net=Flatten()(net)
       net = tf.keras.layers.Dense(flags.label_count,kernel_regularizer=regularizers.L2(0.001))(net)

  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)

  if flags.dt_type=='mel':
    return tf.keras.Model(i.input, net)
  else:
    return tf.keras.Model(input_audio, net)
