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

"""GRU with Mel spectrum and fully connected layers."""
from scipy.signal.filter_design import iircomb
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
    """GRU model parameters."""
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
        '--gru_units',
        type=str,
        default='400',
        help='Output space dimensionality of gru layer',
    )
    parser_nn.add_argument(
        '--return_sequences',
        type=str,
        default='0',
        help='Whether to return the last output in the output sequence,'
        'or the full sequence',
    )
    parser_nn.add_argument(
        '--stateful',
        type=int,
        default='1',
        help='If True, the last state for each sample at index i'
        'in a batch will be used as initial state for the sample '
        'of index i in the following batch',
    )
    parser_nn.add_argument(
        '--dropout1',
        type=float,
        default=0.1,
        help='Percentage of data dropped',
    )
    parser_nn.add_argument(
        '--units1',
        type=str,
        default='128,256',
        help='Number of units in the last set of hidden layers',
    )
    parser_nn.add_argument(
        '--act1',
        type=str,
        default="'linear','relu'",
        help='Activation function of the last set of hidden layers',
    )


def model(flags):
    """Gated Recurrent Unit(GRU) model.

    It is based on paper
    Convolutional Recurrent Neural Networks for Small-Footprint Keyword Spotting
    https://arxiv.org/pdf/1703.05390.pdf (with no conv layer)
    Model topology is similar with "Hello Edge: Keyword Spotting on
    Microcontrollers" https://arxiv.org/pdf/1711.07128.pdf
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
    if flags.dt_type=='mfcc':
        if flags.preprocess == 'raw':
        # it is a self contained model, user need to feed raw audio only
            net = speech_features.SpeechFeatures(
                speech_features.SpeechFeatures.get_params(flags))(net)
        input_audio = tf.keras.layers.Input(shape=(16000,),batch_size=128)
        net = input_audio
    elif flags.dt_type=='mel':
        input_audio=(1,16000)
        i=get_melspectrogram_layer(input_shape=input_audio,
                                    n_mels=40,
                                    pad_end=True,
                                    n_fft=1024,
                                    hop_length=400,
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
        # net=L.Lambda(lambda q:K.expand_dims(q,axis=2),name='expand_midle_dim')(net)
    
    # for units, return_sequences in zip(utils.parse(flags.gru_units), utils.parse(flags.return_sequences)):
    #   net = GRU(units=units, return_sequences=return_sequences, stateful=flags.stateful)(net)
    if flags.att_type!= 'None':     
        # shape = net.shape
        # net = tf.keras.layers.Reshape(
        #     (-1, shape[2]*shape[3]), name='reshape2')(net)
        # value=net
        gru = L.Bidirectional(
            GRU(100, return_sequences=True,recurrent_dropout=0.4,dropout=0.3,kernel_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay)), name="bi_gru_0")(net)
        (gru, forward_h, backward_h) = L.Bidirectional(
            GRU(100, return_sequences=True, return_state=True,recurrent_dropout=0.4,dropout=0.3,kernel_regularizer=tf.keras.regularizers.l2(flags.l2_weight_decay)), name="bi_gru_1")(gru)
        
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
            # net=tf.keras.layers.Dropout(flags.dropout)(net)
            # net=Flatten()(net)

    # net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

    # for units, activation in zip(
    #         utils.parse(flags.units1), utils.parse(flags.act1)):
    #     net = tf.keras.layers.Dense(units=units, activation=activation)(net)

    net = tf.keras.layers.Dense(units=flags.label_count)(net)
    if flags.return_softmax:
        net = tf.keras.layers.Activation('softmax')(net)

    if flags.dt_type=='mel':
         return tf.keras.Model(i.input, net)
    else:
          return tf.keras.Model(input_audio, net)
