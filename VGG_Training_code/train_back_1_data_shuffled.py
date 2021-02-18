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
#SGD_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
SGD_optimizer = tf.keras.optimizers.Adagrad(lr=0.01, decay=0)
#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():
    #back_layer = load_model("../../pretrained_model/back_layers.h5") # pooling 4
back_layer = load_model("/media/2/Network/pretrained_model/back_layers_19~.h5")
"""
    back_layer.layers[0].trainable = True
    for i in range(1,8):
        back_layer.layers[i].trainable = False
"""
back_layer.layers[1].trainable = True
for i in range(2,4):
    back_layer.layers[i].trainable = False
back_layer.compile(optimizer=SGD_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
back_layer.summary()

def packet_drop(arr):
    index = randrange(256)
    arr[:,:,index:(index+2)] = 0
def error_injection(data):
    for img in tqdm(data):
        packet_drop(img)
        
f = open("result_bacK_layer_1_training.txt","w")
f.close()
#path = "/media/2/Network/extracted_feature/whole_shuffle/" # pooling 4
path = "/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error"
for i in tqdm(range(60)):
    print("------ 0~400000 loading!! -------")
    print("np loading...")
    #val_data = np.load("../validation_features.npy")
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
    #val_data = np.load("../extracted_feature/whole_shuffle/validation_features.npy")
    #val_label = np.load("../extracted_feature/whole_shuffle/validation_label.npy")
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
    # val_data = np.load("../extracted_feature/whole_shuffle/validation_features.npy")
    # val_label = np.load("../extracted_feature/whole_shuffle/validation_label.npy")
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
    f = open("result_bacK_layer_1_training.txt","a")
    print(i,"th Loss of the model is - ", back_layer.evaluate(test_data,test_label),file=f)
    f.close()
    # print(i,"th Accuracy of the model is - ", back_layer.evaluate(test_data, test_label)[1]*100, "%")
    del test_data,test_label
    if i % 10 == 0 or i == 59:
        back_layer.save("/media/3/Network/retrain_model_dir/pooling4/1_layer/"+str(i)+"_back_layer_1_training.h5")
    gc.collect()
gc.collect()

