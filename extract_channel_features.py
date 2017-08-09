# -*- coding: utf-8 -*-
"""
Created on Tue May 23 02:11:49 2017

@author: Anoop
"""

#extracting features
#HOG, LUV, GRAD MAG, RGB, HSV


#import files for caluclating feature descriptors

from skimage.transform import resize
from skimage.io import imread
from sklearn.externals import joblib

#timport files to read file names
import argparse as ap
import glob
import os
from config import *

from gradient import *

#from stubs import generate
from vectors import extract

import random
from config import *

from read_write import *
 
CHANNEL_COUNT = 16
UNIQUE_FEATURE_COUNT = feature_count
 
MAX_SIZE = 15
MIN_SIZE = 3
COLLISION_LIMIT = 500 
 
def generate(seed_value, uniq_features):
     '''
     Generates feature stubs
     
     :param seed_value: seed value for the random generation. May be used for
     reproducible feature stubs
     :param uniq_features: number of unique features to be generated
     :return generated feature stub
     '''
     
         
     feature_dict = {}
     failures = 0
     features = 0
     
     while True :
         ch = random.randint(0, CHANNEL_COUNT - 1)
         he = random.randint(5, 25)
         wi = random.randint(5, 30)
         p_x = random.randint(0, min_wdw_size[0] - wi)  #100
         p_y = random.randint(0, min_wdw_size[1] - he)  #80
         
         tup = (ch, p_x, p_y, he, wi)
         
         if not feature_dict.get(tup):
             feature_dict[tup] = True
             features += 1
         else:
             failures += 1
             
         if failures == COLLISION_LIMIT:
             raise Exception('Collision limit reached')
         elif features == uniq_features:
             break
     
#     print feature_dict.keys()
     return feature_dict.keys()
    
    
#############################################################################################

def compute_features(im, feature_stubs):
    chans = get_channels(im)
    
    integral_channels = []
    for chan in chans:
        integral_channels.append(cv2.integral(chan))
#        print integral_channels[len(integral_channels)-1]
    
    feature_vals = []
    
    
    feature_vals.append(extract(integral_channels, feature_stubs))
    
    
          
    return feature_vals




########################################################################################################

if __name__ == "__main__":
    
    
    #if  directoris don't exit, create them
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0]) 
        
        
    print "calculating the descriptors for the positive samples and saving them"
    
    #generate features
    feature_stubs = generate(seed_value=1234, uniq_features=feature_count)
    
    fds = []  
    labels = []
    
    pos_image_count = 0
    #for im_path in glob.glob(os.path.join(pos_im_path, "*.png" or "*.jpg" or "*.jpeg" or "*.PNG")):
    for im_path in os.listdir(pos_im_path):
        im_orig = imread(pos_im_path+im_path, as_grey= False)
        # image shape in order width, height and color
        im = resize(im_orig, (min_wdw_size[1],min_wdw_size[0])) #min_wdw_size = scale*[50, 40]
        im = cv2.convertScaleAbs(im)
        pos_image_count += 1
        print "positive image: " , pos_image_count
        if des_type == "ICF":
            fd = compute_features(im, feature_stubs)
            fds.append(fd)
            labels.append(1)

        

#######################################################################################   
    print "Calculating the descriptors for negative samples and saving them"
    neg_image_count = 0
    for im_path in glob.glob(os.path.join(neg_im_path, "*.png")):
        im_orig = imread(im_path, as_grey= False)
        im = resize(im_orig, (min_wdw_size[1],min_wdw_size[0]))
        im = cv2.convertScaleAbs(im)
        
        neg_image_count += 1
        print "negative image: " , neg_image_count
        if des_type == "ICF":
            fd = compute_features(im, feature_stubs)
            fds.append(fd)
            labels.append(0)

    
    print "Completed calculating features from training images"
    
    
    ##############################
    #save feature stubs and feature_vectors and labels
    write_stub(feature_stubs, feature_stubs_filename)
    
    print "feature stubs are stored in {}".format(feature_stubs_filename)
    
    pickle_features(fds, feature_vectors_filename)
        
    print "feature_vectors are stored in {}".format(feature_vectors_filename)       
            
    pickle_labels(labels, labels_filename)       
            
    print "labels are stored in {}".format(labels_filename)

          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            



