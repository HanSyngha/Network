import random
import os,gc
import PIL
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

back_layers = load_model("../original_models/original_back_layers_model.h5")


for Feature_map_index in range(0,512):
    print("=============== [",Feature_map_index,"] Feature_map error processing ================")
    print("Make training copy")
    training_feature_with_error = training_feature.copy()
    print("Insert Error to training set")
    training_feature_with_error[:,:,:,Feature_map_index] = 0
    print("Error Inserted")
    gc.collect()

    print("Make validation copy")
    validation_feature_with_error = validation_feature.copy()
    print("Insert Error to validation set")
    validation_feature_with_error[:,:,:,Feature_map_index] = 0
    print("Error Inserted")
    gc.collect()

    print(training_feature_with_error.shape,validation_feature_with_error.shape)

    print("Load model")
    back_layers.load_weights("../original_models/original_back_layers_weights.h5")
    back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    print("Model ready")

    print("Training start")
    for Epoch in range(1,11):
        print("[ ",Feature_map_index," ] Feature_map error with [ ",Epoch," ] Epoches testing")
        back_layers.fit(training_feature_with_error,training_label,batch_size=64,epochs=1,validation_data = (validation_feature_with_error,validation_label),validation_batch_size=64) 
        gc.collect()
        file_name = "./single_feature_map_error_weights/"+str(Feature_map_index)+"_feature_map_error_"+str(Epoch)+".h5"
        print("saving[",file_name,"]")
        back_layers.save(file_name)
        print("Deleting model")
        gc.collect()
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


