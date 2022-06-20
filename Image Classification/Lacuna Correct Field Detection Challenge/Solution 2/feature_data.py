import numpy as np
import cv2
import os
from skimage import io, color
import pandas as pd

from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

import coords

from scipy import ndimage


CONST_X = 10.986328125 / 2
CONST_Y = 10.985731758 / 2

W = 80

a = np.arange(W)
a = a[:, np.newaxis]
o = np.ones((1,W))

iY = np.matmul(a,o)
iX = iY.T


def get_mask(x,y,sharpness=0.7):
    ''' gaussian mask distance from target point '''
    Z =  np.exp(-sharpness*((iX-x)**2 + (iY-y)**2))
    return Z




def image_features(id, folder_start=''):

    folders = [
        'planet-jun17/',
        'planet-dec17/',
        'planet-jun18/',
        'planet-dec18/']


    X = []
    for folder in folders:
        I =(io.imread(folder_start+folder+id+'.png'))

        sobelx = cv2.Sobel(I.astype(float)/255,cv2.CV_64F,1,0,ksize=1)
        sobely = cv2.Sobel(I.astype(float)/255,cv2.CV_64F,0,1,ksize=1)

        E = np.sum(sobelx**2 + sobely**2, axis=2, keepdims=True)
        I = color.rgb2lab(I).astype(float)/255

        X.append(E)
        X.append(I)

    return np.dstack(X)


def get_features(dataframes=[], folder_prefixes=[], is_train=False):
    result = {}

    X = []
    y = []
    cs = []
    ws = []
    Ds = []
    tis = []
    tcis = []
    ccis = []
    tws = []


    for kk in range(len(dataframes)):
        df = dataframes[kk]
        folder = folder_prefixes[kk]
        
        for k in range(df.shape[0]):
            r = df.loc[k]

            imid = r['ID']
            imid = imid.split('_')[1]

            I = image_features(imid, folder_start=folder)

            ws.append([ I.shape[1], I.shape[0] ])    

            I, cx0, cy0 = coords.crop(I)


            icx = (ws[-1][0]-1)/2.0 - cx0
            icy = (ws[-1][1]-1)/2.0 - cy0
            
            D =  ( (CONST_X*(iX-icx)/(ws[-1][0]-1))**2
                 + (CONST_Y*(iY-icy)/(ws[-1][1]-1))**2 )**0.5
    

            Ds.append(D)

            
            ccis.append( [icx, icy] )
            cs.append( [cx0, cy0] )
            X.append( I )

            if is_train:

                xi, yi = coords.to_image_space(r['x'], r['y'], ws[-1][0], ws[-1][1])

                tws.append([r['x'], r['y']])
                tis.append([xi, yi])

                xi -= cx0
                yi -= cy0

                tcis.append([xi,yi])
                
                Z = get_mask(xi, yi)
                y.append( Z )


    result['X'] = np.array(X)
    result['widths'] = np.array(ws)
    result['crops'] = np.array(cs)
    result['center_distances'] = np.array(Ds)
    result['center_cropped_image_space'] = np.array(ccis)

    if is_train:
        y = np.array(y)
        result['target_world_space'] = np.array(tws)
        result['target_image_space'] = np.array(tis)
        result['target_cropped_image_space'] = np.array(tcis)
        result['target_masks'] = np.array(y)

    return result




