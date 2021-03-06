from random import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
import time
import datetime 

tf.random.set_seed(960312)
num_of_test = 15
number_of_packet_to_drop = int(sys.argv[1])
drop_num = int(sys.argv[1])
path="/media/2/Network/extracted_feature/whole_shuffle_to_19/"
eval_result_file =  "./result/result_back_1_adam_"+str(drop_num)+"_packet.txt"
back_layer_file = "/media/2/Network/pretrained_model/back_layers_19~.h5"


f = open(eval_result_file,"w")
f.close()
#train_data = np.load(path+"train_feature_"+str(drop_num)+".npy",mmap_mode="r")
#train_label = np.load(path+"train_label_"+str(drop_num)+".npy",mmap_mode="r")
#test_data = np.load(path+"testing_features_"+str(drop_num)+".npy",mmap_mode="r")
#test_label = np.load(path+"testing_label_"+str(drop_num)+".npy",mmap_mode="r")
print("np loading finish")


def packet_drop(arr):
    index = randrange(65-number_of_packet_to_drop)
    arr[:,:,index*8:(index*8+8*number_of_packet_to_drop)] = 0
def error_injection(data):
    for img in data:
        packet_drop(img)

def evaluate(num_of_test,drop_number):
    f = open(eval_result_file,"a")
    f.close()
    back_layer = load_model(back_layer_file) 
    optimizer = tf.keras.optimizers.Adam(lr=5e-5,beta_1=0.9,beta_2=0.999,epsilon=None, decay=1e-3,amsgrad=True)
    back_layer.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    f = open(eval_result_file,"a")
    print("number of packet drop : {}".format(drop_number),file=f)
    f.close()
    for i in range(num_of_test):
        test_data = np.load(path+"testing_features.npy") # original not packet dropped
        test_label = np.load(path+"testing_label.npy")
        error_injection(test_data)             # error injection 
        test_result =back_layer.evaluate(test_data,test_label)
        f = open(eval_result_file,"a")
        print(drop_num,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100),file=f)
        f.close()
        print(drop_num,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100))
        gc.collect()
    
#f = open("result_back_1_adadelta.txt","w")

gc.collect()
"""
val_lr = [5e-05]
val_decay = [1e-3, 1e-6]
val_amsgrad = [False,True]
"""
for i in range(1,drop_num+1):
    evaluate(num_of_test,i)
