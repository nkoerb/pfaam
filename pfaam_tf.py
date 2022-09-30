
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import  Activation,  Lambda,  Reshape
from tensorflow.keras import backend as K
import tensorflow as tf

def PfAAM(x):
    keep = x
    channel_act = GlobalAveragePooling2D()(x)
    spatial_act = Lambda(AveragePoolChannels)(x)
    y = Lambda(MatMul)([spatial_act,channel_act])
    y = Activation("sigmoid")(y)
    res = multiply([keep,y])
    return res
    
def PfAAM_MAX(x):
    keep = x
    channel_act = GlobalMaxPooling2D()(x)
    spatial_act = Lambda(MaxPoolChannels)(x)
    y = Lambda(MatMul)([spatial_act,channel_act])
    y = Activation("sigmoid")(y)
    res = multiply([keep,y])
    return res

   
def MatMul(x):
    y = x[0]
    z = x[1]
    z = Reshape((1,1,z.shape[1]))(z)
    return y * z

def MaxPoolChannels(x):
    return K.max(x, axis=-1, keepdims=True)

def AveragePoolChannels(x):
    return K.mean(x, axis=-1, keepdims= True)


