import os,gc,time
import sys
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np
from keras import layers
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
from keras import layers
from keras.models import load_model
from keras.backend import clear_session
from keras.preprocessing.image import ImageDataGenerator

num = sys.argv[1]


#Optimizer
sgd_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
feature_map_per_packet = 8
Packet_number_to_erase = 1
#Loading Datas
label = np.load("/media/0/Network/VGG16/datasets/testing_label.npy")
print("Test data is ready")

#Loading Original Model
original_back_layers = load_model("/media/0/Network/VGG16/pretrained_model/fc123_model.h5")
original_back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
print("Original Model is ready")

gc.collect()



    #Error insertiong
test_data_with_error = np.load("/media/0/Network/VGG16/datasets/testing_features.npy")

print("Original with error")
loss, acc = original_back_layers.evaluate(test_data_with_error,label)

    #Loading Retrained Model
for Epoch in range(1,21):
    clear_session()
    file_name = "/media/1/Network/VGG16/weights/global/"+num+"/"+str(Epoch)+".h5"
    print("load ",file_name)
    retrained_back_layers = load_model(file_name)
    loss, acc = retrained_back_layers.evaluate(test_data_with_error,label)
    clear_session()
    retrained_back_layers = None
    del retrained_back_layers
    file_name = None
    del file_name
    gc.collect()
        




