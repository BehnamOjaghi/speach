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

"""Mobilenet - reduced version of keras/applications/mobilenet.py."""
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
    """Mobilenet model parameters.

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
        type=int,
        default=32,
        help='Number of filters in the first conv',
    )
    parser_nn.add_argument(
        '--cnn1_kernel_size',
        type=str,
        default='(3,1)',
        help='Kernel size of the first conv',
    )
    parser_nn.add_argument(
        '--cnn1_strides',
        type=str,
        default='(2,2)',
        help='Strides of the first conv',
    )
    parser_nn.add_argument(
        '--ds_kernel_size',
        type=str,
        default='(3,1),(3,1),(3,1),(3,1)',
        help='Kernel sizes of depthwise_conv_blocks',
    )
    parser_nn.add_argument(
        '--ds_strides',
        type=str,
        default='(2,2),(2,2),(1,1),(1,1)',
        help='Strides of depthwise_conv_blocks',
    )
    parser_nn.add_argument(
        '--cnn_filters',
        type=str,
        default='32,64,128,128',
        help='Number of filters in depthwise_conv_blocks',
    )
    parser_nn.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Percentage of data dropped',
    )
    parser_nn.add_argument(
        '--heads',
        type=int,
        default=4,
        help='Number of heads in multihead attention',
    )
    parser_nn.add_argument(
        '--bn_scale',
        type=int,
        default=1,
        help='If True, multiply by gamma. If False, gamma is not used. '
        'When the next layer is linear (also e.g. nn.relu), this can be disabled'
        'since the scaling will be done by the next layer.',
    )


def model(flags):
    """Mobilenet model.

    It is based on paper:
    MobileNets: Efficient Convolutional Neural Networks for
       Mobile Vision Applications https://arxiv.org/abs/1704.04861
    It is applied on sequence in time, so only 1D filters applied
    Args:
      flags: data/model parameters

    Returns:
      Keras model for training
    """
    # input_audio = tf.keras.layers.Input(
    #     shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
    #     batch_size=flags.batch_size)
    input_audio = tf.keras.layers.Input(shape=(16000,))
    net = input_audio
    if flags.dt_type == 'mfcc':
        if flags.preprocess == 'raw':
          # it is a self contained model, user need to feed raw audio only
            net = speech_features.SpeechFeatures(
                speech_features.SpeechFeatures.get_params(flags))(
                    net)

# channels_first
# channels_last
    elif flags.dt_type == 'mel':
        input_audio = (1, 16000)
        i = get_melspectrogram_layer(input_shape=input_audio,
                                     n_mels=80,
                                     pad_end=True,
                                     n_fft=1024,
                                     win_length=512,
                                     hop_length=128,  # 8ms
                                     mel_f_max=8000,
                                     mel_f_min=40.0,
                                     sample_rate=16000,
                                     return_decibel=True,
                                     input_data_format='channels_first',
                                     output_data_format='channels_last')
        net = L.Lambda(lambda q: K.squeeze(q, -1),
                       name='squeeze_last_dim1')(i.output)

        net = SpecAugment()(net,training=True)
        shape = net.shape
        # net = tf.keras.layers.Reshape(
        #     (shape[1], 1, shape[2]), name='reshape1')(net)
        net = L.Lambda(lambda q: K.expand_dims(q, 2),
                       name='expand_middle_dim1')(net)
        # net=L.Lambda(lambda q:K.expand_dims(q,axis=2),name='expand_midle_dim')(net)
        # net=LayerNormalization(axis=0,name='batch_norm')(net)
      #   net=tf.keras.layers.Reshape(net,(-1,net.shape[1],net.shape[3],net.shape[2]))
      #   net=tf.keras.backend.permute_dimensions(net,[0,1,3,2])
      #   net=LayerNormalization(axis=2,name='batch_norm')(i.output)
      #   net=LayerNormalization(axis=2,name='batch_norm')(i.output)

    # [batch, time, feature]
    if flags.dt_type == 'mfcc':
        net = tf.keras.backend.expand_dims(net, axis=2)
        # [batch, time,1, feature]

    # it is convolutional block
    net = tf.keras.layers.Conv2D(
        filters=flags.cnn1_filters,
        kernel_size=utils.parse(flags.cnn1_kernel_size),
        padding='valid',
        # kernel_regularizer=regularizers.l2(0.001),
        use_bias=False,
        strides=utils.parse(flags.cnn1_strides))(
            net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.ReLU(6.)(net)
    # [batch, time, feature, filters]

    for kernel_size, strides, filters in zip(
            utils.parse(flags.ds_kernel_size), utils.parse(flags.ds_strides),
            utils.parse(flags.cnn_filters)):
        # it is depthwise convolutional block
        net = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            padding='same' if strides == (1, 1) else 'valid',
            depth_multiplier=1,
            # kernel_regularizer=regularizers.l2(0.001),
            strides=strides,
            use_bias=False)(
                net)
        net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
        net = tf.keras.layers.ReLU(6.,)(net)

        net = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            # kernel_initializer='he_normal',
            # kernel_regularizer=regularizers.l2(0.001),
            strides=(1, 1))(net)
        net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
        net = tf.keras.layers.ReLU(6.)(net)
        # [batch, time, feature, filters]

    if flags.att_type != 'None':
        shape = net.shape
        net = tf.keras.layers.Reshape(
            (-1, shape[2]*shape[3]), name='reshape2')(net)
        # value=net
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
        # for _ in range(2):
        #     net = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, unroll=True,
        #     dropout=0.4,recurrent_dropout=0.3,kernel_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay),
        #         bias_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay)))(net)
        # net = L.GRU(units=150, return_sequences=True, recurrent_dropout=0.2, dropout=0.4, unroll=True, kernel_regularizer=tf.keras.regularizers.l2(
        #     flags.l2_weight_decay), bias_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay))(net)

    if flags.att_type == 'self_att':
        # net = MySelfAttention2(output_dim=128)([q_v, k_v, k_v])
        net = ScaledDotProductAttention()([q_v, k_v, k_v])
    elif flags.att_type == 'multi_head':
        shape = net.shape
        net = MultiHeadAttention(head_num=2)(net)
        net = tf.reshape(net, shape=(-1, shape[1], net.shape[2]))
        # net = MyMultiHeadAttention(
        #     output_dim=net.shape[2]//2+1, num_head=3)(net)
        #  net=MultiHeadAttention(d_model=128,num_heads=4)(net,k=net,q=net,mask=None)
    elif flags.att_type == 'att':
        # net=Attention(units=2*net.shape[1])(net)
        net = attention()(net)
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

    if flags.att_type != "None":
        net = tf.keras.layers.Dropout(flags.dropout)(net)
        net = Flatten()(net)
    else:
        # [batch, filters]
        net = tf.keras.layers.Dropout(flags.dropout)(net)
    net = tf.keras.layers.Dense(12)(net)
    net = tf.keras.layers.Activation('softmax')(net)
    # [batch, label_count]
    if flags.dt_type == 'mel':
        return tf.keras.Model(i.input, net)
    else:
        return tf.keras.Model(input_audio, net)
