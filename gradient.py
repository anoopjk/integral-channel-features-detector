# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:57:12 2017

@author: Anoop
"""

#computing channel

import cv2
import numpy as np

GRAD_DDEPTH = cv2.CV_16S  #word size for gradient channels
#central degrees for gradient oriented channels
ORIENTATION_DEGREES = [15, 45, 75, 105, 135, 165]
#Bin size for orientation channels
ORIENTATION_BIN = 14

def oriented_gradient(grad_x, grad_y, degree, bin_size):
    
    '''
    Returns the oriented gradient channel
    :param grad_x: Gradient computed only for X axis
    :param grad_y: Gradient computed only for Y axis
    :param degree: Degree of the edge to be calculated
    :param bin_size: Degree margin for which the edges to be calculated
    
    For example, if degree is '30' and bin size is '10', this routine  computes
    edges for the degree interval 20 to 40.
    
    '''
    
    #asserting the shapes of x and y gradients being equal 
    assert grad_x.shape == grad_y.shape
    
    lower_bound = degree - bin_size
    upper_bound = degree + bin_size
    
    rows, cols = grad_x.shape
    
    oriented = np.zeros((rows, cols), np.uint8)
    
    
    for i in xrange(rows):
        for j in xrange(cols):
            e_x = grad_x.item(i,j)
            e_y = grad_y.item(i,j)
            
            d = cv2.fastAtan2(e_y, e_x)
            
            if lower_bound < d < upper_bound:
                oriented.itemset((i,j), 255)
    
    return oriented
    
    
def get_channels(image):
    '''
    calculates and returns in total 16 channels
        - 6 gradient orientation channels
        - gradient magnitude
        - L, U and V channels of cieLUV
		- R, G and B channels of RGB
		- H, S and V channels of HSV
    in this order
    '''
    
    channels = [None for i in range(16)]

    
    assert len(ORIENTATION_DEGREES) == 6
    assert min(ORIENTATION_DEGREES) - ORIENTATION_BIN > 0
    assert max(ORIENTATION_DEGREES) + ORIENTATION_BIN < 180
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (3,3), 0)
    
    gradient_x = cv2.Sobel(gray_image, GRAD_DDEPTH, 1, 0)
    gx_scaled = cv2.convertScaleAbs(gradient_x)
    
    gradient_y = cv2.Sobel(gray_image, GRAD_DDEPTH, 0, 1)
    gy_scaled = cv2.convertScaleAbs(gradient_y)
    
    magnitude = cv2.addWeighted(gx_scaled, 0.5, gy_scaled, 0.5, 0)
    
    channels[6] = magnitude
    
    for i, deg in enumerate(ORIENTATION_DEGREES):
        orie = oriented_gradient(gradient_x, gradient_y, deg, ORIENTATION_BIN)
        orie = cv2.medianBlur(orie, 3)
        orie = cv2.bitwise_and(orie, magnitude)
        
        channels[i] = orie
    
    #adding the  color channels in the order RGB
    #first converting image from bgr to rgb and then adding each channel    
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.split(rgb)             #or image[:,:,::-1]
    for i, cha in enumerate(rgb):
        channels[i+7] = cha
     
     
    #addign the luv channels in the order luv
     #first converting image to luv and then adding each channel
     
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    luv = cv2.split(luv)
    
    for i, cha in enumerate(luv):
        channels[i+10] = cha
        
        
    #adding HSV channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = cv2.split(hsv)
    
    for i, cha in enumerate(hsv):
        channels[i+13] = cha
 
    return channels
    

def get_integral_channels(channels):
    '''
    Transforms the computed channels into integral channels
    :param channels: Regular channels
    :return: Integral channels
    '''
    
    int_chn = map(lambda it: cv2.integral(it), channels)
    return int_chn
    
if __name__ == '__main__':
    img = cv2.imread('C:/Users/Anoop/Desktop/job_applications/IFM/testing/roi.png')
    
    chans = get_channels(img)
    
    integral = cv2.integral(chans[0])
    print integral
    
    
    
    
 
    
    
    
    
    
    
    

    
                
        
    
    
    