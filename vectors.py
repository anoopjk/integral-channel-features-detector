# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:24:10 2017

@author: Anoop
"""

# computing , serialization and deserialization of the  dataset formed from vector stubs
#and training images

import sys

def extract(integral_channels, feature_stubs):
    '''
    calculates the feature values
    :param integral channels: Integral channels of an image
    :param feature_stubs: Feature stubs.
    '''
    
    features = []
    for stub in feature_stubs:
        #print stub
        
        ch, p_x, p_y, he, wi = stub
        
        ch = integral_channels[ch]
        #print ch
        rect1 = ch.item(p_y, p_x)
        rect2 = ch.item(p_y, p_x + wi)
        rect3 = ch.item(p_y+he, p_x)
        rect4 = ch.item(p_y+he, p_x+wi)
        
        fea = rect4 + rect1 - rect2 - rect3
        features.append(fea)
        
        
    return features
        
        
