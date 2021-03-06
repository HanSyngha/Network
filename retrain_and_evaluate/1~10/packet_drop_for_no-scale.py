from random import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import PIL
import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
import math
import sys
import gc
from tqdm import tqdm

tf.random.set_seed(960312)

number_of_packet_to_drop = int(sys.argv[1])
drop_num = number_of_packet_to_drop
print(drop_num,"/64 packet lost")
path = "/media/2/Network/extracted_feature/whole_shuffle_to_19_no-rescale/"
save_path = path+"with_"+str(drop_num)+"_packet_error/"
if os.path.isdir(save_path) is False:
    os.mkdir(save_path)

def packet_drop(arr):
    global number_of_packet_to_drop
    #print(arr.shape)
    index = randrange(65-number_of_packet_to_drop)
    arr[:,:,index*8:(index*8+8*number_of_packet_to_drop)] = 0
def error_injection(data):
    for img in tqdm(data):
        #print(img.shape)
        packet_drop(img)

file_list = os.listdir(path)
train_features = []
train_label = []
testing_features = []
testing_label = []
val_features = []
val_label = []
for item in file_list:
    #print(item)
    if "train_feature" in item:
        #print(item)
        train_features.append(item)
    elif "train_label" in item:
        train_label.append(item)
    elif "test_label" in item:
        testing_label.append(item)
    elif "test_feature" in item:
        testing_features.append(item)
    elif "val_feature" in item:
        val_features.append(item)
    elif "val_label" in item:
        val_label.append(item)
def concatenate(items,save_name):
    global path 
    for index,item in enumerate(items):
        print(index,item)
        if index == 0:
            a = np.load(path+item)#,mmap_mode="r")
            if "feature" in save_name :
                error_injection(a)
            print("===",a.shape)
        else :
            tmp = np.load(path+item)#,mmap_mode="r")
            if "feature" in save_name :
                error_injection(tmp)
            print("===",tmp.shape)
            a = np.concatenate((a,tmp),axis=0)
            print(a.shape)
    np.save(save_path+save_name,a)
    print(save_name+".npy saved, shape :",a.shape)
    del a
    gc.collect()


concatenate(train_label,"train_label_"+str(drop_num)) 
concatenate(train_features,"train_feature_"+str(drop_num))

concatenate(testing_features,"test_feature_"+str(drop_num))
concatenate(testing_label,"test_label_"+str(drop_num))

concatenate(val_features,"val_feature_"+str(drop_num))
concatenate(val_label,"val_label_"+str(drop_num))

import os
os._exit(00)
