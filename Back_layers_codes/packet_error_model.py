import random
import os,gc
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
import math
from keras.backend import clear_session
from keras.models import load_model

#Global parameters
sgd_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
Batch_size = 64
feature_map_per_packet = 8

#Get Training Data
training_feature = np.load("../datasets/training_features.npy")
print("Training Feature",training_feature.shape)
training_label = np.load("../datasets/training_label.npy")
print("Training Label",training_label.shape)
validation_feature = np.load("../datasets/validation_features.npy")
print("Validation Feature",validation_feature.shape)
validation_label = np.load("../datasets/validation_label.npy")
print("Validation Label",validation_label.shape)
print("Datas are Ready")

print("Making back_layers")
original_model = VGG16()
back_layers = load_model("../original_models/original_back_layers_model.h5")

for Packet_number_to_erase in range(1,11):
  for Packet_index in range(0,(65-Packet_number_to_erase)):
      print("=============== [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"] Packet error processing ================")
      print("Make training copy")
      training_feature_with_error = training_feature.copy()
      print("Insert Error to training set")
      training_feature_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
      print("Error Inserted")
      gc.collect()
  
      print("Make validation copy")
      validation_feature_with_error = validation_feature.copy()
      print("Insert Error to validation set")
      validation_feature_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
      print("Error Inserted")
      gc.collect()
  
      print(training_feature_with_error.shape,validation_feature_with_error.shape)
  
      print("Initializing Model")
      back_layers.load_weights("../original_models/original_back_layers_weights.h5")
      back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
  
      print("Training start")
      for Epoch in range(1,11):
          print("[ ",Packet_index+1,":",Packet_index+Packet_number_to_erase," ] Packet error with [ ",Epoch," ] Epochs training")
          back_layers.fit(training_feature_with_error,training_label,batch_size=64,epochs=1,validation_data = (validation_feature_with_error,validation_label),validation_batch_size=64)
          gc.collect()
          file_name = "../Back_layers_weights/"+str(Packet_number_to_erase)+"_packet_error_weights/"+str(Packet_index+1)+"_Packet_error_"+str(Epoch)+".h5"
          print("saving [",file_name,"]")
          back_layers.save(file_name)
      print("Training Finish!")
  
      print("Freeing Datasets")
      clear_session()
      gc.collect()
      training_feature_with_error = None
      del training_feature_with_error
      validation_feature_with_error = None
      del validation_feature_with_error
      clear_session()
      gc.collect()




