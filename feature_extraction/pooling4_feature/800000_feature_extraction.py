import random
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from keras import Sequential
import math
import sys
import gc

tf.random.set_seed(960312)
#Spliting VGG

pretrained_model = load_model("/media/2/Network/pretrained_model/vgg_model.h5")
front_layers = Sequential([layer for layer in pretrained_model.layers[:15]])
#front_layers.summary()

train_datagen = ImageDataGenerator(rescale=1./255)

train_data_generator = train_datagen.flow_from_directory(
        '../Imagenet_dup/train/',
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle = True)

Number_of_train_data = len(train_data_generator.filenames)


train_features = []
train_label = []


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    for i in tqdm(range(00000,Number_of_train_data)):
        try:
            train_features.extend(front_layers.predict(train_data_generator[i][0]))
            train_label.extend(train_data_generator[i][1])
        except:
            continue
        if i % 1000 == 0 and i != 800000:
            gc.collect()
train_features = np.array(train_features)
train_label = np.array(train_label)


np.save("/media/2/Network/extracted_feature/whole_shuffle/train_features_400000",train_features)
np.save("/media/2/Network/extracted_feature/whole_shuffle/train_label_400000",train_label)

print("np.save finished")
