import random
import os,gc
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
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
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
feature_map_per_packet = 8
Packet_number_to_erase = 1

#Get Training Data
training_label = np.load("/media/0/Network/VGG16/datasets/training_label.npy")
print("Training Label",training_label.shape)
validation_label = np.load("/media/0/Network/VGG16/datasets/validation_label.npy")
print("Validation Label",validation_label.shape)
print("Datas are Ready")

print("Making back_layers")

for Packet_index in range(60,(65-Packet_number_to_erase)):
    if os.path.exists("/media/1/Network/VGG16/weights/fc/2/"+str(Packet_index+1)+"_10.h5"):
        continue
    print("=============== [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"] Packet error processing ================")
    training_feature_with_error = np.load("/media/0/Network/VGG16/datasets/training_features.npy")
    validation_feature_with_error = np.load("/media/0/Network/VGG16/datasets/validation_features.npy")

    print("Insert Error to training set")
    training_feature_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
    print("Error Inserted")
    gc.collect()
  
    print("Insert Error to validation set")
    validation_feature_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
    print("Error Inserted")
    gc.collect()
  
    print(training_feature_with_error.shape,validation_feature_with_error.shape)
  
    print("Initializing Model")
    with strategy.scope():
        back_layers_o = load_model("/media/0/Network/VGG16/pretrained_model/fc123_model.h5")
        back_layers_o.layers[1].trainable = False
        back_layers_o.layers[3].trainable = False
        back_layers_o.load_weights("/media/0/Network/VGG16/pretrained_model/fc123_weights.h5")
        back_layers = Sequential()
        for layers in back_layers_o.layers:
            back_layers.add(layers)
        sgd_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
        back_layers.build([None,7,7,512])
        back_layers.summary()

    print("Training start")
    for Epoch in range(1,11):
        file_name = "/media/1/Network/VGG16/weights/fc/2/"+str(Packet_index+1)+"_"+str(Epoch)+".h5"
        if os.path.exists(file_name):
            continue
        print("[ ",Packet_index+1,":",Packet_index+Packet_number_to_erase," ] Packet error with [ ",Epoch," ] Epochs training")
        back_layers.fit(training_feature_with_error,training_label,batch_size=256,epochs=1,validation_data = (validation_feature_with_error,validation_label),validation_batch_size=256)
        gc.collect()
        print("saving [",file_name,"]")
        back_layers.save(file_name)
    print("Training Finish!")
  
    print("Freeing Datasets")
    back_layers = None
    del back_layers
    back_layers_o = None
    del back_layers_o
    training_feature_with_error = None
    del training_feature_with_error
    validation_feature_with_error = None
    del validation_feature_with_error
    gc.collect()




