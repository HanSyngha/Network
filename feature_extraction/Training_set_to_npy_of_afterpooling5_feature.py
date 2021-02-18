import random
import os
import PIL
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
import math
import sys

#Global parameters
sgd_optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#Defined Functions
def Packet_error_insert(Feature_map, Starting_index_to_erase):
    batch, height, width, ch = Feature_map.shape
    Num_of_FM = math.ceil(Packet_size /(height*width))
    Maximum_feature = min (512, Starting_index_to_erase-1+Num_of_FM)
    Feature_map[batch-1,:,:,Starting_index_to_erase-1:Maximum_feature] = 0
    return Feature_map
    
def delete_list(list_to_delete):
    list_to_delete = None
    del list_to_delete
    
def Feature_map_error_insert(Feature_map, Feature_map_index_to_erase):
    batch, height, width, ch = Feature_map.shape
    Feature_map[batch-1,:,:,Feature_map_index_to_erase] = 0
    return Feature_map


train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
training_data_generator = train_datagen.flow_from_directory(
        '/data1/datasets/Imagenet/train',
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        subset='training')
validation_data_generator = train_datagen.flow_from_directory(
        '/data1/datasets/Imagenet/train',
        target_size=(224,224),
        batch_size=1,
        class_mode='categorical',
        subset='validation')
        
Number_of_training_datas = len(training_data_generator.filenames)
Number_of_validation_datas = len(validation_data_generator.filenames)

#Spliting VGG
pretrained_model = VGG16()
pretrained_model.load_weights("./original_models/original_vgg_model.h5")
front_layers = Sequential([layer for layer in pretrained_model.layers[:19]])

features = []
label = []
for i in tqdm(range(0,Number_of_training_datas)):
    features.extend(front_layers.predict(training_data_generator[i][0]))
    label.extend(training_data_generator[i][1])
features = np.array(features)
label = np.array(label)
print("save")
np.save("training_features",features)
np.save("training_label",label)

features = []
label = []
for i in tqdm(range(0,Number_of_validation_datas)):
    features.extend(front_layers.predict(validation_data_generator[i][0]))
    label.extend(validation_data_generator[i][1])
features = np.array(features)
label = np.array(label)


print("save")
np.save("validation_features",features)
np.save("validation_label",label)

    
