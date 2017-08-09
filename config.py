'''
Set the config variable.
'''

import ConfigParser as cp
import json

config = cp.RawConfigParser()
config.read('config.cfg')

min_wdw_size = json.loads(config.get("ICF","min_wdw_size"))
step_size = json.loads(config.get("ICF", "step_size"))
feature_count = json.loads(config.get("ICF", "FEATURE_NUM"))
n_estimators = json.loads(config.get("ICF", "n_estimators"))
max_depth = json.loads(config.get("ICF", "max_depth"))

im = config.get("paths", "im")
pos_im_path =  config.get("paths", "pos_im_path")
neg_im_path =  config.get("paths", "neg_im_path")
des_type = config.get("descriptor", "des_type")
clf_type = config.get("classifier", "clf_type")
feature_stubs_filename = config.get("paths", "feature_stubs_filename")
feature_vectors_filename = config.get("paths", "feature_vectors_filename")
labels_filename= config.get("paths", "labels_filename")

model_path = config.get("paths", "model_path")
threshold = config.getfloat("nms", "threshold")
