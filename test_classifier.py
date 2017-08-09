# -*- coding: utf-8 -*-
"""
Created on Tue May 23 02:45:02 2017

@author: Anoop
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 14 14:19:35 2017

@author: Anoop
"""

#testing the classifier code
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.transform import rescale
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

import cv2
import argparse as ap
from nms import nms
from config import*
import numpy as np

from gradient import *
from vectors import extract
from extract_channel_features import compute_features
from read_write import *
from fastNms import non_max_suppression_fast

def flatten_list(l):
    flattened = []
    for sublist in l:
        for val in sublist:
            flattened.append(val)
    
    return flattened
    
def display_detections(clone ,nmsdetections):
 
    if not len(nmsdetections) == 0:
        for  i in  range(nmsdetections.shape[0]):
            # Draw the detections
            x_tl = nmsdetections[i, 0]
            y_tl = nmsdetections[i, 1]
            x_br = nmsdetections[i, 2]
            y_br = nmsdetections[i, 3]
        
            cv2.rectangle(clone, (x_tl, y_tl), (x_br,y_br), (0, 0, 0), thickness=2)   
        
        cv2.imshow("Final Detections after applying NMS", clone)
        cv2.imwrite('nms_output.png', clone)
        cv2.waitKey()
        cv2.destroyAllWindows() 
            
                 

def sliding_window(image, window_size, step_size):
    '''
    This function return a patch of the input image of size equal to window_size. 
    the first image returned top-left coordinates(0,0) and are incremented in both
    x and y directions by the 'step_size' supplied.
    the input parameters are- 
    *'image' - Input image
    *'window_size' -Size of the Sliding Window
    *'step_size' - incremented size of the window
    
    THe function returns a tuple -
    (x,y, im_window)
    where 
    *x is top-left x coordinate
    *y is top-left y coordinate
    *im_window is the slding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield(x,y, image[y:y + window_size[1], x:x + window_size[0]])
            

if __name__ == "__main__":

    
    #read the image
    #im = imread(args["image"], as_grey = True)
    im = imread(im, as_grey= False)
    im = rescale(im, 1/3.2)
    #min_wdw_size = (100, 80)
    step_size = step_size
    downscale = 1.25
    visualize_det =  True                          #args['visualize']
    
    #load feature stubs
    feature_stubs = read_stub(feature_stubs_filename)
    #Load the classifier
    #clf = unpickle_model(model_path)
    clf = joblib.load(model_path)
    print clf
    
    #list to store the detections
    detections1 = []
    detections2 = []
    #the current scale of the image
    scale = 0
    #Downscale the image and iterate
    scale_counter = 0
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        print "entered gaussian pyramid loop"
        print im_scaled.shape
        im_scaled = cv2.convertScaleAbs(im_scaled)
        #convert image back to uint8 ( that's how I trained the features)
        scale_counter += 1
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print scale_counter
        #this list contains detections at the current scale
        cd =[]
        #if the width or height of the scaled image is less than the width or height
        #of the window, then break from the loop.
        if not scale_counter < 2:
            print "scale_counter has hit 2"
            break
            
#        if im_scaled.shape[0] < min_wdw_size[1] or im_scaled.shape[1] < min_wdw_size[0] :
#            print "im_scaled size is less than that of min window" 
#            break
        print min_wdw_size
        for (x,y, im_window) in sliding_window(im_scaled, min_wdw_size, step_size):
#                print "entered window loop"
#                print "im_window.shape", im_window.shape
#                print "min_wdw_size",min_wdw_size
                if im_window.shape[0] != min_wdw_size[1] or im_window.shape[1] != min_wdw_size[0]:
                    continue
                
                fd = compute_features(im_window, feature_stubs)
                fd = flatten_list(fd)
                fd = np.array(fd)
                fd = fd.reshape(1,-1)
                #print fd
                pred = clf.predict(fd)
#                print pred
#                print clf.decision_function(fd)
                if pred == 1 and clf.decision_function(fd) > 0.4:
                    print "Detection::Location ->({}, {})".format(x,y)
                    print "Scale -> {} | Confidence Score {} \n".format(scale, clf.decision_function(fd))
                    
                    detections1.append((x,y, clf.decision_function(fd),
                                       int(min_wdw_size[0]*(downscale**scale)),
                                       int(min_wdw_size[1]*(downscale**scale))))
                                       
                    #detections2.append((x,y, x+int(min_wdw_size[0]*(downscale**scale)), y + int(min_wdw_size[1]*(downscale**scale))))
                                       
                    cd.append(detections1[-1])
                    
                #if visualize is set to true, display the working
                #of the sliding window
                if visualize_det:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _ in cd:
                        #Draw the detections at this scale
                        cv2.rectangle(clone, (x1,y1), (x1 + im_window.shape[1], y1 + im_window.shape[0]),
                                      (0,0,0), thickness=2)
                        cv2.rectangle(clone, (x,y), (x+ im_window.shape[1], y+ im_window.shape[0]),
                                      (255, 255, 255), thickness= 2)
                        cv2.imshow("sliding window in progress", clone)
                        cv2.waitKey(30)
        #move the next scale
        scale += 1
              
    #Display the results before performing NMS
    clone = im.copy()
    clone2 = im.copy()
    for (x_tl, y_tl, _, w, h) in detections1:
        # Draw the detections
        cv2.rectangle(clone2, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
        
    cv2.imshow("Raw Detections before NMS", im)
    cv2.waitKey(1)
    
    #perform Non Maxima Suppression
    detections1 = nms(detections1, threshold=0.5)
    
    # Display the results after performing NMS
    
#    nmsdetections = non_max_suppression_fast(np.asarray(detections2,dtype = float), threshold)
#    display_detections(clone ,nmsdetections)
    
    for (x_tl, y_tl, _, w, h) in detections1:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
           
    cv2.imshow("Final Detections after applying NMS", clone)
    cv2.imwrite("output.jpg", clone)
    cv2.waitKey()
    cv2.destroyAllWindows() 
                    
                    
                    
    
    
