import tensorflow as tf
from keras import initializers
from keras import activations
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
 
class MySelfAttention(Layer):
        
    def __init__(self,output_dim,kernel_initializer='glorot_uniform',**kwargs):
        self.output_dim=output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MySelfAttention,self).__init__(**kwargs)
        
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
class MySelfAttention2(Layer):
        
    def __init__(self,output_dim,kernel_initializer='glorot_uniform',**kwargs):
        self.output_dim=output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MySelfAttention2,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.W=self.add_weight(name='W',
             shape=(3,input_shape[2][2],self.output_dim),
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
        q_v,k_v,v_v=x
        q=K.dot(q_v,self.W[0])
        k=K.dot(k_v,self.W[1])
        v=K.dot(v_v,self.W[2])
        #print('q_shape:'+str(q.shape))
        e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))#Transpose k and multiply it with q
        e=e/(self.output_dim**0.5)
        e=K.softmax(e)
        o=K.batch_dot(e,v)
        return o
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)
class SWSA(Layer):
        
    def __init__(self,output_dim,kernel_initializer='glorot_uniform',**kwargs):
        self.output_dim=output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(SWSA,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.W=self.add_weight(name='W',
             shape=(1,input_shape[2],self.output_dim),
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
        k=K.dot(x,self.W[0])
        v=K.dot(x,self.W[0])
        #print('q_shape:'+str(q.shape))
        e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))#Transpose k and multiply it with q
        e=e/(self.output_dim**0.5)
        e=K.softmax(e)
        o=K.batch_dot(e,v)
        return o
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)
 
class MyMultiHeadAttention(Layer):
    def __init__(self,output_dim,num_head,kernel_initializer='glorot_uniform',**kwargs):
        self.output_dim=output_dim
        self.num_head=num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MyMultiHeadAttention,self).__init__(**kwargs)

    def get_config(self):
        config=super().get_config().copy()
        config.update({
            'output_dim':self.output_dim,
            'num_head':self.num_head,
            'kernel_initializer':self.kernel_initializer
        })
        return config
        
    def build(self,input_shape):
        self.W=self.add_weight(name='W',
           shape=(self.num_head,3,input_shape[2],self.output_dim),
           initializer=self.kernel_initializer,
           trainable=True)
        self.Wo=self.add_weight(name='Wo',
           shape=(self.num_head*self.output_dim,self.output_dim),
           initializer=self.kernel_initializer,
           trainable=True)
        self.built = True
        
    def call(self,x):
        q=K.dot(x,self.W[0,0])
        k=K.dot(x,self.W[0,1])
        v=K.dot(x,self.W[0,2])
        e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))#Transpose k and multiply it with q
        e=e/(self.output_dim**0.5)
        e=K.softmax(e)
        outputs=K.batch_dot(e,v)
        for i in range(1,self.W.shape[0]):
            q=K.dot(x,self.W[i,0])
            k=K.dot(x,self.W[i,1])
            v=K.dot(x,self.W[i,2])
            #print('q_shape:'+str(q.shape))
            e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))#Transpose k and multiply it with q
            e=e/(self.output_dim**0.5)
            e=K.softmax(e)
            #print('e_shape:'+str(e.shape))
            o=K.batch_dot(e,v)
            outputs=K.concatenate([outputs,o])
        z=K.dot(outputs,self.Wo)
        return z
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        self.units=units
        self.W = Dense(units, use_bias=True, bias_initializer='random_normal', kernel_initializer='random_normal')
        self.V = Dense(1, kernel_initializer='random_normal', use_bias=False)
        super(Attention, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            # 'W': self.W,
            # 'V': self.V,
            'units':self.units
        })
        return config

    def call(self, values):
        score = self.V(tf.nn.tanh(self.W(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
