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
#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():
    #back_layer = load_model("../../pretrained_model/back_layers.h5")

def packet_drop(arr):
    index = randrange(64)
    arr[:,:,index*8:(index*8+8)] = 0
def error_injection(data):
    for img in tqdm(data):
        packet_drop(img)
        
path = "/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error_whole/"
train_data = np.load(path+"whole_feature.npy",mmap_mode="r")
train_label = np.load(path+"whole_label.npy",mmap_mode="r")
test_data = np.load(path+"testing_features.npy",mmap_mode="r")
test_label = np.load(path+"testing_label.npy",mmap_mode="r")
print("np loading finish")

#f = open("result_back_1_adadelta.txt","w")
f = open("result_back_1_rmsprop.txt","w")
f.close()
for val_lr in [0.1,0.005,0.001,0.0001]:
    print("lr :",val_lr)
    back_layer = load_model("/media/2/Network/pretrained_model/back_layers_19~.h5")
    back_layer.layers[1].trainable = True
    for i in range(2,4):
        back_layer.layers[i].trainable = False
    #optimizer = tf.keras.optimizers.Adagrad(lr=val_lr, decay=0)
    optimizer = tf.keras.optimizers.RMSprop(lr=val_lr,rho=0.9,epsilon=None, decay=0)
    back_layer.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    back_layer.summary()
    for i in tqdm(range(10)):
        back_layer.fit(train_data,train_label, validation_split=0.2, epochs=1, verbose=1,batch_size=256)
    
        #f = open("result_back_1_adadelta.txt","w")
        f = open("result_back_1_rmsprop.txt","a")
        test_result =back_layer.evaluate(test_data,test_label)
        print(i,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100),file=f)
        f.close()
        print(i,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100))
    
        if i != 0 and i %3 == 0:
            back_layer.save("/media/3/Network/retrain_model_dir/pooling4/1_layer/"+"RMSprop"+str(i)+"_back_layer_1_training.h5")
    gc.collect()
gc.collect()

