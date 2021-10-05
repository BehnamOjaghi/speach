import tensorflow as tf
from keras import initializers
from keras import activations
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.keras.backend import backend, batch_dot

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self,output_dim,kernel_initializer='glorot_uniform',**kwargs):
        self.output_dim=output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(SelfAttention,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.W=self.add_weight(name='W',
             shape=(3,input_shape[2],self.output_dim),
             initializer=self.kernel_initializer,
             trainable=True)
        self.built = True
    def get_config(self):
        config=super().get_config().copy()
        config.update({
            'output_dim':self.output_dim,
            'kernel_initializer':self.kernel_initializer
        })
        return config

    def call(self,x):
        q=K.dot(x,self.W[0])
        k=K.dot(x,self.W[1])
        v=K.dot(x,self.W[2])
        #print('q_shape:'+str(q.shape))
        e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))#Transpose k and multiply it with q
        e=e/(self.output_dim**0.5)
        e=K.softmax(e)
        o=K.batch_dot(e,v)
        return o
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weight.
         q, k, v must have matching front dimensions.
         k, v must have a matching penultimate dimension, for example: seq_len_k = seq_len_v.
         Although the mask has different shapes according to its type (filled or forward),
         But the mask must be able to perform broadcast conversion in order to sum.

         Parameters:
         q: requested shape == (..., seq_len_q, depth)
         k: The shape of the primary key == (..., seq_len_k, depth)
         v: The shape of the value == (..., seq_len_v, depth_v)
         mask: Float tensor whose shape can be converted to
                     (..., seq_len_q, seq_len_k). The default is None.

         return value: 
         Output, attention weight
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Zoom matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k), so the score
    # Add is equal to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class cMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model,num_heads):
        super(cMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # self.batch_size=batch_size

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config=super().get_config().copy()
        config.update({
            'num_heads':self.num_heads,
            'd_model':self.d_model
            # 'batch_size':self.batch_size
        })
        return config
        
    def split_heads(self, x, batch_size):
        """
                 Split the last dimension into (num_heads, depth).
                 Transpose the result so that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        # batch_size=100
        # batch_size=self.batch_size

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output



# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # values shape == (batch_size, max_len, hidden size)

    # we are doing this to broadcast addition along the time axis to calculate the score
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector

# https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
class attention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()