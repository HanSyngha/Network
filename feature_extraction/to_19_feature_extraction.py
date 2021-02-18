import random
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.optimizers import SGD
from keras import Sequential
import math
import sys
import gc

print("random seed!!!")

tf.random.set_seed(960312)

#Spliting VGG
#pretrained_model = load_model("/media/0/Network/VGG16/pretrained_model/vgg_model.h5")
#mirrored_strategy=tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():
print("loading model!!!!!")
pretrained_model=vgg16.VGG16() # 모델 불러오기
print("model slicing")
front_layers = Sequential([layer for layer in pretrained_model.layers[:19]]) # max_pooling 5까지 진행
front_layers.summary()

def packet_drop(arr):
    index=randrange(64)
    arr[:,:,8*index:(8*index+8)]=0
def error_injection(data):
    for img in tqdm(data):
        packet_drop(img)


###########################  시작!!!!!!!!!!!!!!!!!!

print("training start!!!!")
train_datagen=ImageDataGenerator(validation_split=0.2, rescale=1./255)
#train_datagen=ImageDataGenerator(rescale=1/255)

train_data_generator = train_datagen.flow_from_directory(
        '/media/3/Imagenet_subset/train',
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        subset='training',
        shuffle = True)



Number_of_train_data = len(train_data_generator.filenames)
#Number_of_train_data = int(Number_of_train_data)
print("# of train_data :",Number_of_train_data)
train_features = []
train_label = []



for i in range(0,Number_of_train_data):
    try:
        train_features.extend(front_layers.predict(train_data_generator[i][0]))
        train_label.extend(train_data_generator[i][1])
    except:
        continue


"""
for i in tqdm(range(0, Number_of_train_data)):
    try:
        train_features.extend(front_layers.predict(train_data_generator[i][0]))
        train_label.extend(train_data_generator[i][1])
    except:
        continue
"""

gc.collect()

print("for loop finish")
train_features = np.array(train_features)
train_label = np.array(train_label)
#error_injection(train_features)
print("np conversion finish")
np.save("/media/2/Network/extracted_feature/divided_400000_shuffle_with_error/train_features",train_features)
np.save("/media/2/Network/extracted_feature/divided_400000_shuffle_with_error/train_label",train_label)
print("----- feature extract finished -----")
print(train_features.shape,train_label.shape)

train_features = None
del train_features
train_label = None
del train_label
Number_of_train_data=None
del Number_of_train_data
train_data_generator=None
del train_data_generator
gc.collect()



print("validation start!!!!!!!!!!!")

validation_data_generator = train_datagen.flow_from_directory(
        '/media/3/Imagenet_subset/train',
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        subset='validation',
        shuffle = True)

Number_of_validation_data = len(validation_data_generator.filenames)
#Number_of_validation_data = int(Number_of_validation_data)
print("# of train_data :",Number_of_validation_data)
validation_features = []
validation_label = []



for i in range(0,Number_of_validation_data):
    try:
        validation_features.extend(front_layers.predict(validation_data_generator[i][0]))
        validation_label.extend(validation_data_generator[i][1])
    except:
        continue

"""
for i in tqdm(range(0, Number_of_validation_data)):
    try:
        validation_features.extend(front_layers.predict(validation_data_generator[i][0]))
        validation_label.extend(validation_data_generator[i][1])
    except:
        continue
"""

gc.collect()

print("for loop finish")
validation_features = np.array(validation_features)
validation_label = np.array(validation_label)
#error_injection(validation_features)
print("np conversion finish")
np.save("/media/2/Network/extracted_feature/divided_400000_shuffle_with_error/validation_features",validation_features)
np.save("/media/2/Network/extracted_feature/divided_400000_shuffle_with_error/validation_label",validation_label)
print("----- feature extract finished -----")
print(validation_features.shape,validation_label.shape)

validation_features = None
del validation_features
validation_label = None
del validation_label
Number_of_validation_data=None
del Number_of_validation_data
validation_data_generator=None
del validation_data_generator
gc.collect()


"""


#for i in tqdm(range(0,Number_of_train_data)):
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    for i in tqdm(range(0,400000)):
        try:
            train_features.extend(front_layers.predict(train_data_generator[i][0]))
            train_label.extend(train_data_generator[i][1])
        except:
            continue
        if i% 1000 :
            gc.collect()
gc.collect()

print("for loop finish")
train_features = np.array(train_features)
train_label = np.array(train_label)
error_injection(train_features)
print("np conversion finish")
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error/train_features_0",train_features)
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error/train_label_0",train_label)
print("----- 0~400000 feature extract finished -----")
print(train_features.shape,train_label.shape)

train_features = None
del train_features
train_label = None
del train_label
gc.collect()


5)
train_features = []
train_label = []
#for i in tqdm(range(0,Number_of_train_data)):
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    for i in tqdm(range(400000,800000)):
        try:
            train_features.extend(front_layers.predict(train_data_generator[i][0]))
            train_label.extend(train_data_generator[i][1])
        except:
            continue
        if i% 1000 :
            gc.collect()
gc.collect()

print("for loop finish")
train_features = np.array(train_features)
train_label = np.array(train_label)
error_injection(train_features)
print("np conversion finish")
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error/train_features_400000",train_features)
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error/train_label_400000",train_label)
print("----- 400000~800000 feature extract finished -----")
print(train_features.shape,train_label.shape)

train_features = None
del train_features
train_label = None
del train_label
gc.collect()



train_features = []
train_label = []

#for i in tqdm(range(0,Number_of_train_data)):
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    for i in tqdm(range(800000,Number_of_train_data)):
        try:
            train_features.extend(front_layers.predict(train_data_generator[i][0]))
            train_label.extend(train_data_generator[i][1])
        except:
            continue
        if i% 1000 :
            gc.collect()
gc.collect()

print("for loop finish")
train_features = np.array(train_features)
train_label = np.array(train_label)
error_injection(train_features)
print("np conversion finish")

np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error/train_features_800000",train_features)
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/with_error/train_label_800000",train_label)
print("----- 8000000~1281167 feature extract finished -----")
print(train_features.shape,train_label.shape)

train_features = None
del train_features
train_label = None
del train_label
gc.collect()


"""

############# test 시작######################

print("test start!!!!!!!!!")

test_datagen = ImageDataGenerator(rescale=1./255)

test_data_generator = test_datagen.flow_from_directory(
        '/media/3/Imagenet_subset/test',
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle = True)

Number_of_test_datas = len(test_data_generator.filenames)
#Number_of_test_datas = int(Number_of_test_datas)
print("Number of test datas", Number_of_test_datas)
features = []
label = []
print("yes!!!!!!!!!!!!!!!")


"""
for i in tqdm(range(0,Number_of_test_datas+1)):
    try:
        features.extend(front_layers.predict(train_data_generator[i*3000:(i*3000)+3000][0]))
        label.extend(train_data_generator[i*3000 : (i*3000)+3000][1])
    except:
        continue

"""

for i in tqdm(range(0, Number_of_test_datas)):
    try:
        features.extend(front_layers.predict(test_data_generator[i][0]))
        label.extend(test_data_generator[i][1])
    except:
        continue


gc.collect()

features = np.array(features)
label = np.array(label)
print("----- test extract finished -----")
print(features.shape,label.shape)
error_injection(features)
print("save")
np.save("/media/2/Network/extracted_feature/divided_400000_shuffle_with_error/testing_features",features)
np.save("/media/2/Network/extracted_feature/divided_400000_shuffle_with_error/testing_label",label)


