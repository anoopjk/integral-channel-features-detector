# -*- coding: utf-8 -*-
"""
Created on Wed May 24 18:06:57 2017

@author: Anoop
"""

import pickle


#writing feature_stubs and feautures to files

def write_stub(feature_stubs, filename):
    
    #each stub contains channel no., x_tl, y_tl, width and height
    
    with open(filename, 'w') as fout:
        for stub in feature_stubs:
            feature_stub = '%s %s %s %s %s\n' % (stub[0], stub[1], stub[2], stub[3], stub[4])
            fout.write(feature_stub)
            
            
            
def read_stub(filename):
    
    with open(filename, 'r') as fin:
        data = fin.read()
        data = data.split('\n')
        data.pop()
        
        data = map(lambda it: it.split(' '), data)
        data = map(lambda it: map(lambda el: int(el), it), data)
        
    return data
    
#################################################################################    

def pickle_features(feature_vectors, filename):
    with open(filename, "wb") as fp:  #pickle
        pickle.dump(feature_vectors, fp)
        
    
def unpickle_features( filename):
    with open(filename, "rb") as fp: #unpickle
        feature_vectors = pickle.load(fp)
        
    return feature_vectors
        
#################################################################################

def pickle_labels(labels, filename):
    with open(filename, "wb") as fp: #pickle
        pickle.dump(labels, fp)
        
def unpickle_labels(filename):
    with open(filename, "rb") as fp: #unpickle
        labels = pickle.load(fp)
        
    return labels
    
#################################################################################
def pickle_model(model, filename):
    with open(filename, "wb") as fp: #pickle
        pickle.dump(model, fp)
        
def unpickle_model(filename):
    with open(filename, "rb") as fp: #unpickle
        model = pickle.load(fp)
        
    return model
        

