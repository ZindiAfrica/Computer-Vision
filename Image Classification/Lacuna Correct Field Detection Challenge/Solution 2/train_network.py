import numpy as np
import os
import pandas as pd

from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

import coords
import feature_data

from scipy import ndimage
import gc

import argparse 


SEED = 7

N_FOLD = 4
N_EPOCH = 25
BATCH_SIZE = 8

W = 80

IMAGE_FOLDERS = ['', 'extra_train-']
DATAFRAME_PATHS = ['train-unique.csv', 'extra_train.csv']
OUTPUT_PATH = ''
NETWORK_NAME = 'final_network'


np.random.seed(SEED)
tf.random.set_seed(SEED)

DATAFRAMES = [pd.read_csv(file) for file in DATAFRAME_PATHS]

class DataGen(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=8, is_validation=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.is_validation = is_validation

    def __len__(self):
        return int(self.X.shape[0]//self.batch_size)

    def __getitem__(self, i):
        indxs = np.arange(i*self.batch_size, (i+1)*self.batch_size)

        X = self.X[indxs, :, :, :]
        y = self.y[indxs, :, :]


        if not self.is_validation:
            rot    = np.random.randint(4)
            flip = (1 == np.random.randint(2))
            
            for k in range(X.shape[0]):
                X[k,:,:,:] = ndimage.rotate(X[k,:,:,:], rot*90, reshape=False)
                y[k,:,:] =   ndimage.rotate(y[k,:,:]  , rot*90, reshape=False)

            if flip:
                X = X[:,:,::-1,:]
                y = y[:,:,::-1]
        
        return X, y



def antirectified_arcsinhlu(x):
    '''
        Activation function
        Like ELU but with arcsinh in place of exponential
        Antirectifed -> sinhlu(x) and -sinhlu(-x) are concatenated and returned
    '''
    xp = tf.math.maximum( x, 0.0)
    xn = tf.math.maximum(-x, 0.0)

    return tf.concat( (xp - tf.math.asinh(xn),
                       xn - tf.math.asinh(xp)), axis=3)



def spatial_binary_crossentropy_loss(y_true, y_pred):
    '''
        Training loss
        0 <= y_true < 1
        0 <= y_pred < 1
        sum(y_pred) == 1
        error is minimized when a maximal ammount of field y_pred is
        distributed where y_true is near one. y_pred does not neccessarilly
        have to equal y_true to minimize loss.
    '''
    s = - tf.math.log( tf.math.reduce_sum(
                       tf.math.reduce_sum(
                                         y_true*y_pred
                     , axis=1), axis=1) )
    return s


def spatial_softmax(x):
    '''
        softmax over spatial dimensions
    '''
    m = tf.math.reduce_max(
        tf.math.reduce_max(x,
                           axis=1, keepdims=True),
                           axis=2, keepdims=True)

    x = tf.math.exp( x - m )

    x = ( x
          /
          tf.math.reduce_sum(
          tf.math.reduce_sum(
              x, axis=1, keepdims=True)
               , axis=2, keepdims=True)
          )


    return x

def zero_mask_borders(x, width=80, border_width=14):
    '''
        set spatial borders to zero. Done to prevent network from
        incrementally learning the location of image borders which can lead
        to overfitting.
    '''

    msk = tf.pad(tf.ones( (1,width-2*border_width,
                             width-2*border_width,1)),
                 tf.constant([[0,0], [border_width,border_width],
                                     [border_width,border_width],[0,0]]))
    x = msk*x
    return x

def mirror_padded_conv2d(x, out_dim):

    x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), 'REFLECT')
    x = layers.Conv2D(out_dim, 3, padding="valid")(x)

    return x

def mirror_padded_conv2d_transpose(x, out_dim):

    x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), 'REFLECT')
    x = layers.Conv2DTranspose(out_dim, 3, padding="same")(x)
    x = x[:,1:-1,1:-1,:]

    return x


def get_model():
    inputs = keras.Input(shape=(W,W,16,))

    x = inputs

    past = []

    x = layers.BatchNormalization()(x)

    n_pooling = 2
    filter_nums = [24, 48]


    for k in range(n_pooling):
        
        #x = tf.pad(x, tf.constant([[0,0], [1,1], [1,1],[0,0]]), 'REFLECT')
        #x = layers.Conv2D(filter_nums[k], 3, padding="valid")(x)
        x = mirror_padded_conv2d(x, filter_nums[k])

        past.append(x)

        x = layers.BatchNormalization()(x)
        x = antirectified_arcsinhlu(x)
        x = layers.MaxPooling2D(2)(x)

        
    x = mirror_padded_conv2d(x, 2*filter_nums[-1])
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(.5)(x)
    x = antirectified_arcsinhlu(x)



    x = mirror_padded_conv2d_transpose(x, filter_nums[-1])
    x = layers.BatchNormalization()(x)

    for k in range(n_pooling-1, -1, -1):
        x = antirectified_arcsinhlu(x)

        x = mirror_padded_conv2d_transpose(x, filter_nums[k])
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)

        p = past[-1]
        past=past[:-1]

        p = layers.BatchNormalization()(p)

        x = tf.concat((x, p),axis=3)


    x = mirror_padded_conv2d_transpose(x, 16)
    x = antirectified_arcsinhlu(x)


    x = layers.BatchNormalization()(x)


    x1 = layers.Dense(8, activation='linear')(x)
    x1 = tf.math.asinh(x1)
    x2 = layers.Dense(4, activation='linear')(x)
    x2 = antirectified_arcsinhlu(x2)

    x = tf.concat((x1, x2), axis=3)

    x = layers.Dense(1, activation='linear')(x)

    x = spatial_softmax(x)
    x = zero_mask_borders(x)

    #rescale x to sum to 1 over spatial dimensions
    x = x/tf.math.reduce_sum(
          tf.math.reduce_sum(x,
                             axis=1, keepdims=True),
                             axis=2, keepdims=True)


    model = keras.Model(inputs, x)
    return model


optimizer = keras.optimizers.RMSprop(
    momentum=0.7,

)


data = feature_data.get_features(
    DATAFRAMES, IMAGE_FOLDERS, is_train=True)

X = data['X'] 
y = data['target_masks']

x_std = np.std( data['target_world_space'][:,0] )
y_std = np.std( data['target_world_space'][:,1] )

m_std = (x_std + y_std)/2.0 

np.save(f'{OUTPUT_PATH}{NETWORK_NAME}_target_std.npy', m_std)


N = X.shape[0]
a = np.arange(N)
for k in range(N_FOLD):
    bt =  0 < (a <  ((k+0)*N)//N_FOLD)+(a >= ((k+1)*N)//N_FOLD)

    dgt = DataGen(X[bt,:,:,:],y[bt,:,:], batch_size=BATCH_SIZE)

    model = get_model()
    #model.summary()
    model.compile(optimizer=optimizer, loss=spatial_binary_crossentropy_loss)


    history = model.fit(dgt, epochs=N_EPOCH, verbose=2)

    model.save(f'{OUTPUT_PATH}{NETWORK_NAME}_fold_{k}_of_{N_FOLD}')
    
    del model, dgt,  bt 
    gc.collect()
