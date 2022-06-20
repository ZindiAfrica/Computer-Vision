import numpy as np
import os
import pandas as pd

from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

import coords
import feature_data



CONST_X = 10.986328125 / 2
CONST_Y = 10.985731758 / 2

W = 80


N_FOLD = 4

IMAGE_FOLDERS = ['']
DATAFRAME_PATH = 'test.csv'
OUTPUT_PATH = ''
MODEL_PATH = ''
MODEL_NAME = 'final_network'


df = pd.read_csv(DATAFRAME_PATH)


def normal( x, u=0.0, s=1.0 ):
    y = np.exp( -( (x - u)/s )**2/2.0 ) / ( s * np.sqrt( 2 * np.pi ) )

    return y


data = feature_data.get_features(
    [df], IMAGE_FOLDERS, is_train=False)

X = data['X']
Ds = data['center_distances']
cs = data['crops']
ws = data['widths']

# a geometric average is taken over the output of each fold
model_pred = np.ones((X.shape[0],W,W,1))
for k in range(N_FOLD):
    model = keras.models.load_model(f'{MODEL_PATH}{MODEL_NAME}_fold_{k}_of_{N_FOLD}',
                                    custom_objects={'spatial_binary_crossentropy_loss':None})
    model_pred *= model.predict(X)
    
model_pred = model_pred**(1/N_FOLD)

target_std = np.load(f'{MODEL_PATH}{MODEL_NAME}_target_std.npy')

S = []

for k in range(model_pred.shape[0]):

    #pixel world space distances form origin
    D = Ds[k,:,:]

    # gaussian distribution probability of finding target at pixel given
    # distance from origin
    M =  normal( D, s = target_std*2/3.)

    # probability of finding target at pixel given distance from origin and
    # model output
    Z = M*model_pred[k,:,:,0]

    xc, yc = coords.median_centroid(Z)

    # uncrop centroid
    xc += cs[k][0]
    yc += cs[k][1]


    x, y = coords.from_image_space(xc, yc, ws[k][0], ws[k][1])

    r = df.loc[k]

    S.append( [r['ID'], x, y])


S = np.array(S)
S = pd.DataFrame( data=S, columns=['ID', 'x', 'y'] )


S.to_csv(f'{OUTPUT_PATH}solution.csv', index=False)
