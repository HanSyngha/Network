import random
import os,gc,time
import PIL
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'
import tensorflow as tf
import numpy as np
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
import math
from keras.backend import clear_session
from keras.models import load_model

tf.random.set_seed(960312)
#Global parameters
sgd_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
Batch_size = 64


#Get Training Data
while os.path.exists("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/training_features.npy") == False:
    time.sleep(100)
while os.path.exists("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/training_label.npy") == False:
    time.sleep(100)
while os.path.exists("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/validation_features.npy") == False:
    time.sleep(100)
while os.path.exists("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/validation_label.npy") == False:
    tiem.sleep(100)

training_feature = np.load("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/training_features.npy")
print("Training Feature",training_feature.shape)
training_label = np.load("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/training_label.npy")
print("Training Label",training_label.shape)
validation_feature = np.load("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/validation_features.npy")
print("Validation Feature",validation_feature.shape)
validation_label = np.load("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/validation_label.npy")
print("Validation Label",validation_label.shape)
print("Datas are Ready")

print("Making back_layers")
original_model = VGG16()
back_layers = load_model("/media/2/Network/pretrained_model/fc123_model.h5")
back_layers.layers[2].trainable = False
back_layers.layers[3].trainable = False
back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])


error_index = 0

for img in tqdm(training_feature):
    img[:,:,error_index:error_index+8] = 0
    error_index = (error_index+8)%512


error_index = 0

for img in tqdm(validation_feature):
    img[:,:,error_index:error_index+8] = 0
    error_index = (error_index+8)%512

print("Error inserted")
print("Training start")

for Epoch in range(1,21):
    print("[",Epoch,"] / [20] Epoch training")
    file_name = "/media/3/Network/weights/global/1/"+str(Epoch)+".h5"
    if os.path.exists(file_name):
        continue
    back_layers.fit(training_feature,training_label,batch_size=64,epochs=1,validation_data = (validation_feature,validation_label),validation_batch_size=64)
    gc.collect()
    print("saving [",file_name,"]")
    back_layers.save(file_name)

print("Training Finish")
clear_session()

