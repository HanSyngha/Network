from random import *
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from keras.optimizers import SGD
from keras import Sequential
import math
import sys
import gc
import PIL
tf.random.set_seed(960312)
def check_valid(gen):
    while True:
        try :
            data,label = gen.__next__()
            yield data,label
        except GeneratorExit:
            print("Exit")
            break
        except :
            print("other exception occur")
            pass
            

train_path = "../../Imagenet_dup/train"
val_path = "../../Imagenet_dup/val"
npy_train_path = "/media/3/Imagenet_npy/train"
npy_val_path = "/media/3/Imagenet_npy/val"

test_datagen = ImageDataGenerator(rescale=1./255)
print("start reading test image file")
test_data_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=(224, 224),
        #batch_size=128,
        batch_size=1,
        class_mode='categorical',
        shuffle = True)

val_feature = []
val_label = []
f = open("tmp.txt","a")
print("val conversion start",len(test_data_generator),file=f)
f.close()
for index,data in enumerate(check_valid(test_data_generator)):
    if index == len(test_data_generator):
        f = open("tmp.txt","a")
        print("end of test generator",file=f)
        f.close()
        break
    img_data,img_label = data[0],data[1]
    val_feature.extend(img_data)
    val_label.extend(img_label)
    if index % 1000 == 0 :
        print(index,end="/")
    if index % 200000 == 0 and index != 0:
        val_feature = np.array(val_feature)
        val_label = np.array(val_label)
        print(val_feature.shape,val_label.shape)
        feature_path = npy_val_path +"/val_data_"+str(index)
        np.save(feature_path,val_feature)
        label_path = npy_val_path +"/val_label_"+str(index)
        np.save(label_path,val_label)
        f = open("tmp.txt","a")
        print(index,"saved",file=f)
        f.close()
        val_feature = list()
        val_label = list()
        gc.collect()
val_feature = np.array(val_feature)
val_label = np.array(val_label)
print(val_feature.shape,val_label.shape)
feature_path = npy_val_path +"/val_data_last"
np.save(feature_path,val_feature)
label_path = npy_val_path +"/val_label_last"
np.save(label_path,val_label)
del val_feature,val_label
gc.collect()


train_datagen = ImageDataGenerator(rescale=1./255)
print("start reading train image file")
train_data_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        #batch_size=128,
        batch_size=1,
        class_mode='categorical',
        shuffle = True)

f = open("tmp.txt","w")
print("train conversion start",len(train_data_generator),file=f)
f.close()
train_feature = []
train_label = []
for index,data in enumerate(check_valid(train_data_generator)):
    if index == len(train_data_generator):
        f = open("tmp.txt","a")
        print("end of generator",file=f)
        f.close()
        break
    img_data,img_label = data[0],data[1]
    train_feature.extend(img_data)
    train_label.extend(img_label)
    if index % 1000 == 0 :
        print(index,end="/")
    #img_class = np.argmax(img_label)
    if index % 200000 == 0 and index != 0:
        train_feature = np.array(train_feature)
        train_label = np.array(train_label)
        print(train_feature.shape,train_label.shape)
        feature_path = npy_train_path +"/train_data_"+str(index)
        np.save(feature_path,train_feature)
        label_path = npy_train_path +"/train_label_"+str(index)
        np.save(label_path,train_label)
        f = open("tmp.txt","a")
        print(index,"saved",file=f)
        f.close()
        train_feature = []
        train_label = []
        gc.collect()
train_feature = np.array(train_feature)
train_label = np.array(train_label)
print(train_feature.shape,train_label.shape)
feature_path = npy_train_path +"/train_data_last"
np.save(feature_path,train_feature)
label_path = npy_train_path +"/train_label_last"
np.save(label_path,train_label)
del train_feature,train_label
gc.collect()
