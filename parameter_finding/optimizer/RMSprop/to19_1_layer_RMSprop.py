from random import *
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
from keras.utils import multi_gpu_model
from keras import Sequential
import math
import sys
import gc


tf.random.set_seed(960312)

number_of_packer_to_drop = 1
path = "/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error_whole/"
training_result_file =  "result_back_1_rmsprop.txt"
back_layer_file = "/media/2/Network/pretrained_model/back_layers_19~.h5"
val_lr = [0.001,0.0005,0.0001]
val_rho = [0.1, 0.5,0.9]
val_decay = [0, 1e-3, 1e-6]


train_data = np.load(path+"whole_feature.npy",mmap_mode="r")
train_label = np.load(path+"whole_label.npy",mmap_mode="r")
test_data = np.load(path+"testing_features.npy",mmap_mode="r")
test_label = np.load(path+"testing_label.npy",mmap_mode="r")
print("np loading finish")

f = open(training_result_file,"w")
f.close()

def packet_drop(arr):
    index = randrange(65-number_of_packer_to_drop)
    arr[:,:,index*8:(index*8+8*number_of_packer_to_drop)] = 0
def error_injection(data):
    for img in tqdm(data):
        packet_drop(img)

def training(lr,rho=0.9,decay=0):
    f = open(training_result_file,"a")
    print("RMSprop option : lr: {}, rho: {},  decay: {}".format(lr,rho,decay),file=f)
    f.close()
    back_layer = load_model(back_layer_file)
    back_layer.layers[1].trainable = True
    for i in range(2,4):
        back_layer.layers[i].trainable = False
    #optimizer = tf.keras.optimizers.Adagrad(lr=val_lr, decay=0)
    optimizer = tf.keras.optimizers.RMSprop(lr=lr,rho=rho,epsilon=None, decay=decay)
    back_layer.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    for i in range(60):
        back_layer.fit(train_data,train_label, validation_split=0.2, epochs=1, verbose=1,batch_size=256)

        #f = open("result_back_1_adadelta.txt","w")
        f = open(training_result_file,"a")
        test_result =back_layer.evaluate(test_data,test_label)
        print(i,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100),file=f)
        f.close()
        print(i,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100))
        if i %10 == 9:
            back_layer.save("/media/3/Network/retrain_model_dir/pooling4/1_layer/"+"RMSprop"+str(i)+"_back_layer_1_training.h5")
        gc.collect()

#f = open("result_back_1_adadelta.txt","w")

gc.collect()
f = open(training_result_file,"a")
print("RMSprop, pooling5 1 layer training of one packet_drop",file=f)
f.close()
"""
val_lr = [0.001,0.0005,0.0001]
val_rho = [0.1, 0.5,0.9]
val_decay = [0, 1e-3, 1e-6]
"""
for lr in val_lr:
    for rho in val_rho:
        for decay in val_decay:
            training(lr,rho,decay)
