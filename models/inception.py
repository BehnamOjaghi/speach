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

"""Inception - reduced version of keras/applications/inception_v3.py ."""
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
  """Inception model parameters.

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
      '--cnn1_filters',
      type=str,
      default='24',
      help='Number of filters in the first conv blocks',
  )
  parser_nn.add_argument(
      '--cnn1_kernel_sizes',
      type=str,
      default='5',
      help='Kernel size in time dim of conv blocks',
  )
  parser_nn.add_argument(
      '--cnn1_strides',
      type=str,
      default='1',
      help='Strides applied in pooling layer in the first conv block',
  )
  parser_nn.add_argument(
      '--cnn2_filters1',
      type=str,
      default='10,10,16',
      help='Number of filters inside of inception block '
      'will be multipled by 4 because of concatenation of 4 branches',
  )
  parser_nn.add_argument(
      '--cnn2_filters2',
      type=str,
      default='10,10,16',
      help='Number of filters inside of inception block '
      'it is used to reduce the dim of cnn2_filters1*4',
  )
  parser_nn.add_argument(
      '--cnn2_kernel_sizes',
      type=str,
      default='5,5,5',
      help='Kernel sizes of conv layers in the inception block',
  )
  parser_nn.add_argument(
      '--cnn2_strides',
      type=str,
      default='2,2,1',
      help='Stride parameter of pooling layer in the inception block',
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.2,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--bn_scale',
      type=int,
      default=0,
      help='If True, multiply by gamma. If False, gamma is not used. '
      'When the next layer is linear (also e.g. nn.relu), this can be disabled'
      'since the scaling will be done by the next layer.',
  )


def model(flags):
  """Inception model.

  It is based on paper:
  Rethinking the Inception Architecture for Computer Vision
      http://arxiv.org/abs/1512.00567
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
#   input_audio = tf.keras.layers.Input(
#       shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
#       batch_size=flags.batch_size)
  
  input_audio = tf.keras.layers.Input(shape=(16000,))
  net = input_audio
  if flags.dt_type=='mfcc':
      if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
        net = speech_features.SpeechFeatures(
            speech_features.SpeechFeatures.get_params(flags))(
                net)

# channels_first
# channels_last
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

  for stride, filters, kernel_size in zip(
      utils.parse(flags.cnn1_strides),
      utils.parse(flags.cnn1_filters),
      utils.parse(flags.cnn1_kernel_sizes)):
    net = utils.conv2d_bn(
        net, filters, (kernel_size, 1), padding='valid', scale=flags.bn_scale)
    if stride > 1:
      net = tf.keras.layers.MaxPooling2D((3, 1), strides=(stride, 1))(net)

  for stride, filters1, filters2, kernel_size in zip(
      utils.parse(flags.cnn2_strides), utils.parse(flags.cnn2_filters1),
      utils.parse(flags.cnn2_filters2), utils.parse(flags.cnn2_kernel_sizes)):

    branch1 = utils.conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)

    branch2 = utils.conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)
    branch2 = utils.conv2d_bn(
        branch2, filters1, (kernel_size, 1), scale=flags.bn_scale)

    branch3 = utils.conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)
    branch3 = utils.conv2d_bn(
        branch3, filters1, (kernel_size, 1), scale=flags.bn_scale)
    branch3 = utils.conv2d_bn(
        branch3, filters1, (kernel_size, 1), scale=flags.bn_scale)

    net = tf.keras.layers.concatenate([branch1, branch2, branch3])
    # [batch, time, 1, filters*4]
    net = utils.conv2d_bn(net, filters2, (1, 1), scale=flags.bn_scale)
    # [batch, time, 1, filters2]

    if stride > 1:
      net = tf.keras.layers.MaxPooling2D((3, 1), strides=(stride, 1))(net)

#   net = tf.keras.layers.GlobalAveragePooling2D()(net)
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

    q_v = tf.reshape(q_v, shape=(-1, 1, q_v.shape[1]))
   
  if flags.att_type=='self_att':
    #   net=MySelfAttention(output_dim=net.shape[2])(net)
     net = ScaledDotProductAttention()([q_v, k_v, k_v])
  elif flags.att_type=='multi_head':
      net=MyMultiHeadAttention(output_dim=net.shape[1],num_head=3)(net)
  elif flags.att_type=='att':
      net=Attention(units=net.shape[1])(net)
  elif flags.att_type == 'midle_att':

      feature_dim = net.shape[-1]
      middle = net.shape[1] // 2  # index of middle point of sequence
      # feature vector at middle point [batch, feature]
      mid_feature = net[:, middle, :]
    # apply one projection layer with the same dim as input feature
      query = tf.keras.layers.Dense(feature_dim)(mid_feature)

    # attention weights [batch, time]
      att_weights = tf.keras.layers.Dot(axes=[1, 2])([query, net])
      att_weights = tf.keras.layers.Softmax(name='attSoftmax')(att_weights)

    # apply attention weights [batch, feature]
      net = tf.keras.layers.Dot(axes=[1, 1])([att_weights, net])
  elif flags.att_type == 'mh_midle_att':
        feature_dim = net.shape[-1]
        middle = net.shape[1] // 2  # index of middle point of sequence

        # feature vector at middle point [batch, feature]
        mid_feature = net[:, middle, :]
        # prepare multihead attention
        multiheads = []
        for _ in range(flags.heads):
            # apply one projection layer with the same dim as input feature
            query = tf.keras.layers.Dense(
                feature_dim,
                kernel_regularizer=tf.keras.regularizers.l2(
                    flags.l2_weight_decay),
                bias_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay))(
                    mid_feature)

            # attention weights [batch, time]
            att_weights = tf.keras.layers.Dot(axes=[1, 2])([query, net])
            att_weights = tf.keras.layers.Softmax()(att_weights)
            # apply attention weights [batch, feature]
            multiheads.append(tf.keras.layers.Dot(
                axes=[1, 1])([att_weights, net]))

        net = tf.keras.layers.concatenate(multiheads)
  else:
        net = tf.keras.layers.GlobalAveragePooling2D()(net)
  
  if flags.att_type !="None":
      net=tf.keras.layers.Dropout(flags.dropout)(net)
      net=Flatten()(net)
  else:
      # [batch, filters*4]
      net = tf.keras.layers.Dropout(flags.dropout)(net)

#   # [batch, filters*4]
#   net = tf.keras.layers.Dropout(flags.dropout)(net)
  net = tf.keras.layers.Dense(flags.label_count,kernel_regularizer=regularizers.l2(l=0.01))(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
    
  if flags.dt_type=='mel':
    return tf.keras.Model(i.input, net)
  else:
    return tf.keras.Model(input_audio, net)
#   return tf.keras.Model(input_audio, net)
