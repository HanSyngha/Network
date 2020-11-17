import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#set GPU number to use 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model

#early_stopping = EarlyStopping(monitor='val_accuracy',min_delta=0,patience=1,verbose=1,mode='max')

training_datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255)

train_generator = training_datagen.flow_from_directory(
        '../datasets/Imagenet/train',
        target_size=(224,224),
        batch_size=64,
        class_mode='categorical'
        subset='training')

validation_generator = validation_datagen.flow_from_directory(
        '../datasets/Imagenet/train',
        target_size=(224,224),
        batch_size=64,
        class_mode='categorical'
        subset='validation')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        '../datasets/Imagenet/val',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')

model = VGG16(weights='imagenet')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(train_generator,batch_size=64,epochs=84,validation_data=test_generator,validation_batch_size=64,callbacks=[early_stopping])
model.fit(train_generator,batch_size=64,epochs=84,validation_data=validation_generator,validation_batch_size=64)
model.evaluate(test_generator)


model.save("../original_models/pre_trained_model.h5")
print("training finished")
