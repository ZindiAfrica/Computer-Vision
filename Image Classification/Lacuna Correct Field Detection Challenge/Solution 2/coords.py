import numpy as np

CONST_X = 10.986328125 / 2
CONST_Y = 10.985731758 / 2

w_cropped = 80


a = np.arange(w_cropped)
a = a[:, np.newaxis]
o = np.ones((1,w_cropped))

iY = np.matmul(a,o)
iX = iY.T


iXz = iX - (w_cropped-1)/2
iYz = iY - (w_cropped-1)/2


def to_image_space(x, y, w, h):
    '''
    converts from geospatial coordinates to image space coordinates
    '''
    ximg = (w-1) * ( 0.5 - x / CONST_X ) 
    yimg = (h-1) * ( 0.5 + y / CONST_Y )

    return ximg,yimg

def from_image_space(ximg, yimg, w, h):
    '''
    converts from image space coordinates to geospatial coordinates
    '''
    
    x = - CONST_X * ( ximg/(w-1) - 0.5 )
    y = + CONST_Y * ( yimg/(h-1) - 0.5 )
    return x,y


def crop(I):
    '''
    crops the image to 80x80 pixels. Returns the cropped image
    and the number of pixels cropped from the left and top
    '''

    cx0 = 1*( ( I.shape[1] - w_cropped + 1 )//2 )
    cx1 = 1*( ( I.shape[1] - w_cropped + 0 )//2 )
    cy0 = 1*( ( I.shape[0] - w_cropped + 1 )//2 )
    cy1 = 1*( ( I.shape[0] - w_cropped + 0 )//2 )    

    I = I[ +cy0:-cy1, +cx0:-cx1, : ]

    return I, cx0, cy0,

'''
def crop_sentinel(I):

    cx0 = 1*( ( I.shape[1] - 40 + 1 )//2 )
    cx1 = 1*( ( I.shape[1] - 40 + 0 )//2 )
    cy0 = 1*( ( I.shape[0] - 40 + 1 )//2 )
    cy1 = 1*( ( I.shape[0] - 40 + 0 )//2 )    


    
    I = I[ +cy0:(I.shape[0]-cy1), +cx0:(I.shape[1]-cx1), : ]

    return I
'''


def centroid(Z):
    x = np.sum( Z*iX ) / np.sum( Z )
    y = np.sum( Z*iY ) / np.sum( Z )
    return x, y

def median_centroid(Z):
    '''
    returns the weighted median centroid for image weights Z
    '''
    x = (w_cropped-1)/2
    y = (w_cropped-1)/2

    for k in range(10):
        W = Z/( ( (x-iX)**2 + (y-iY)**2 + 0.5 )**0.5 )

        x = np.sum( iX*W ) / np.sum( W )
        y = np.sum( iY*W ) / np.sum( W )

    return x, y
        




    
