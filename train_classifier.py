# -*- coding: utf-8 -*-
"""
Created on Tue May 16 01:12:46 2017

@author: Anoop
"""

from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import argparse as ap
import glob
import os
from config import *
from read_write import *


def flatten_list(l):
    flattened = []
    for sublist in l:
        for val in sublist:
            flattened.append(val)
    
    return flattened
        
        
        
if __name__ == "__main__":
    
    
    #read the feature_vectors file
    feature_stubs = read_stub(feature_stubs_filename)
    
    #read the feature_labels file
    labels = unpickle_labels(labels_filename)
    
    #read the feature_vectors file
    fds = unpickle_features(feature_vectors_filename)
        
    fds = flatten_list(fds)
         
    feature_vals = fds
    if clf_type == "LIN_SVM":
        
        clf = LinearSVC(class_weight= 'balanced')
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        # If  directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print "Classifier saved to {}".format(model_path)
    
    
    
    
    elif clf_type == "Adaboost":
        
        dt = DecisionTreeClassifier(max_depth=2) 
        clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
        print "Training a  Adaboost ensemble Classifier"
        clf.fit(fds, labels)
        # If  directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        
        #pickle_model(clf, model_path)
        joblib.dump(clf, model_path)
        print "Classifier saved to {}".format(model_path)    


