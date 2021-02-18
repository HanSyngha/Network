from random import *
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
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

# back layer 전체를 training 시킬것이다!!!!
back_layer = load_model("../../pretrained_model/back_layers.h5")
#for i in range(0,8):
#    back_layer.layers[i].trainable = True
SGD_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#back_layer=multi_gpu_model(back_layer,gpus=2) # 2개의 gpu를 사용한다.
#back_layer.build((None,14,14,512))
back_layer.compile(optimizer=SGD_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
back_layer.summary()

def packet_drop(arr):
    index = randrange(256)
    arr[:,:,index:(index+2)] = 0
def error_injection(data):
    for img in tqdm(data):
        packet_drop(img)

f = open("result_back_layer_whole_training.txt","w")
f.close()
path = "/media/2/Network/extracted_feature/whole_shuffle/"
for i in tqdm(range(60)):
    print("------ 0~400000 loading!! -------")
    print("np loading...")
    #val_data = np.load("../extracted_feature/whole_shuffle/validation_features.npy")
    #val_label = np.load("../extracted_feature/whole_shuffle/validation_label.npy")
    train_data = np.load(path+"train_features_0.npy")
    train_label = np.load(path+"train_label_0.npy")
    print("np loading finish")
    gc.collect()
    #error_injection(val_data)
    error_injection(train_data)
    print("random error injection finished")
    
    # validation split이 0.2니까 training feature의 20%를 validation using
    back_layer.fit(train_data,train_label, validation_split=0.2, epochs=1, verbose=1,batch_size=64)
    #del val_data,val_label
    del train_data,train_label
    gc.collect()
    print("------ 400000~800000 loading!! -------")
    print("np loading...")
    train_data = np.load(path+"train_features_400000.npy")
    train_label = np.load(path+"train_label_400000.npy")
    print("np loading finish")
    gc.collect()
    #error_injection(val_data)
    error_injection(train_data)
    print("random error injection finished")
    
    back_layer.fit(train_data,train_label, validation_split=0.2, epochs=1, verbose=1, batch_size=64)
    #del val_data,val_label
    del train_data,train_label
    gc.collect()
    print("------ 800000~ loading!! -------")
    print("np loading...")
    train_data = np.load(path+"train_features_800000.npy")
    train_label = np.load(path+"train_label_800000.npy")
    print("np loading finish")
    gc.collect()
    # error_injection(val_data)
    error_injection(train_data)
    print("random error injection finished")

    back_layer.fit(train_data,train_label, validation_split=0.2, epochs=1, verbose=1, batch_size=64)
    #del val_data,val_label
    del train_data,train_label
    gc.collect()
    ############ evaluate ####################

    test_data = np.load(path+"testing_features.npy")
    test_label = np.load(path+"testing_label.npy")
    print("test_data loading finish")
    error_injection(test_data)
#    print(i,"th Loss of the model is - ", back_layer.evaluate(test_data,test_label))
    
    f = open("result_back_layer_whole_training.txt","a")
    print(i,"th Loss of the model is - ", back_layer.evaluate(test_data,test_label),file=f)
    f.close()
    # print(i,"th Accuracy of the model is - ", back_layer.evaluate(test_data, test_label)[1]*100, "%")
    del test_data,test_label
    back_layer.save("/media/2/Network/retrain_model_dir/pooling4/"+str(i)+"_back_layer_whole_training")
    gc.collect()
gc.collect()


