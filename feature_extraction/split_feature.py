from random import *
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from PIL import Image
import PIL
import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import SGD
#from keras.utils.training_utils import multi_gpu_model
from keras import Sequential
import math
import sys
import gc
import errno,shutil

import shutil
def dir_maker(path): 
    try: 
        os.mkdir(path)
        print("directory is made",path)
    except OSError as exc:
        #print(exc.errno,)
        if exc.errno != errno.EEXIST:
            raise
        pass
def rm_dir_force(rm_path):
    # it should be tmp folder
    print(rm_path)
    if "tmp" not in rm_path : # for safety
        print("it might not be tmp folder",rm_path)
        raise
    try :
        shutil.rmtree(rm_path)
    except OSError as exc:
        if exc.errno is 2: # file not exist
            pass
        else :
            raise
type_dict = { "train_data":0 ,"train_label":1}
def split_data(data,i,main_path,split_size,type_):
    # after split data , remove original data for memory space
    # stor split_data in path , each split_file size is split_size
    # type_ : 0 - features, 1 - label
    # making tmp array
    type_ = type_dict[type_]
    dir_maker(main_path)
    file_name = str()
    ref_name =  str()
    if type_ is 0:  # feature
        path = os.path.join(main_path,"train_data")
        ref_name = path +"/data_tmp_"
    elif type_ is 1: # label
        path = os.path.join(main_path,"train_label")
        ref_name = path + "/label_tmp_"
    while True:
        # split into one big file into file_size / size
        a = data[(i%8)*split_size:((i+1)%9)*split_size]
        if len(a) is 0:
            print(a.shape,"data end")
            break
        else :
            print(a.shape)
        # no need to insert error
        #if type_ is 0:  # feature
        #    error_injection(a,1)
        file_name = ref_name +str(i)
        dir_maker(path)
        file_path = file_name
        print(file_path)
    # error injection
        np.save(file_path,a)
        del a
        i = i+1
        gc.collect()
    data = None
    del data
    return i
features = ["../../extracted_feature/whole_shuffle/train_features_0.npy",
           "../../extracted_feature/whole_shuffle/train_features_400000.npy",
           "../../extracted_feature/whole_shuffle/train_features_800000.npy"]
labels = ["../../extracted_feature/whole_shuffle/train_label_0.npy",
           "../../extracted_feature/whole_shuffle/train_label_400000.npy",
           "../../extracted_feature/whole_shuffle/train_label_800000.npy"]
path = '../../extracted_feature/tmp'
split_size = 51200
f_index = 0
l_index = 0
for i in range(3):
    print("loading ",features[i])
    train_data = np.load(features[i])
    split_data(train_data,f_index,path,split_size,"train_data")
    del train_data
    gc.collect()

    print("loading ",labels[i])
    train_label = np.load(labels[i])
    split_data(train_label,l_index,path,split_size,"train_label")
    del train_label
    gc.collect()
    f_index = f_index +7
    l_index = l_index +7
for file in os.listdir(path+"/train_label"):
    print(file)
print("==========")
for file in os.listdir(path+"/train_data"):
    print(file)    
