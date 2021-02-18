import random
import os
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
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

"""
이전에 pooling4까지 뽑았던 feature를 이용해서
pooling 4에서 pooling 5사이에서 feature를 뽑는다.
"""


#Spliting VGG
pretrained_model = load_model("../pretrained_model/vgg_model.h5")
front_layers = Sequential([layer for layer in pretrained_model.layers[15:19]]) # max_pooling 4까지 진행
#SGD_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
front_layers.build((None, 14, 14, 512))
#front_layers.compile(optimizer=SGD_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
front_layers.summary()



########################### training 시작!!!!!!!!!!!!!!!!!!
print("----training start!!!!----")
features = np.load("../extracted_feature/whole_shuffle/train_features_0.npy")
label = np.load("../extracted_feature/whole_shuffle/train_label_0.npy")

Number_of_features=len(features)/3000
Number_of_features=int(Number_of_features)
features_predict=[]

# 한번에 GPU에 다 안올라가서 쪼개서 넣기
for i in range(0,Number_of_features+1):
    features_predict.extend(front_layers.predict(features[i*3000:(i*3000)+3000], verbose=1))
        
features_predict=np.array(features_predict)
print("---- features_predict shape")
print(features_predict.shape)

print("---- 0~400000 training save-----")
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/train_features_0",features_predict)
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/train_label_0",label)

features = None
del features
label = None
del label
features_predict=None
del features_predict
gc.collect()


print("----training start!!!!----")
features = np.load("../extracted_feature/whole_shuffle/train_features_400000.npy")
label = np.load("../extracted_feature/whole_shuffle/train_label_400000.npy")

Number_of_features=len(features)/3000
Number_of_features=int(Number_of_features)
features_predict=[]

# 한번에 GPU에 다 안올라가서 쪼개서 넣기
for i in range(0,Number_of_features+1):
    features_predict.extend(front_layers.predict(features[i*3000:(i*3000)+3000], verbose=1))

features_predict=np.array(features_predict)
print("---- features_predict shape")
print(features_predict.shape)

print("---- 0~400000 training save-----")
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/train_features_400000",features_predict)
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/train_label_400000",label)

features = None
del features
label = None
del label
features_predict=None
del features_predict
gc.collect()



print("----training start!!!!----")
features = np.load("../extracted_feature/whole_shuffle/train_features_800000.npy")
label = np.load("../extracted_feature/whole_shuffle/train_label_800000.npy")

Number_of_features=len(features)/3000
Number_of_features=int(Number_of_features)
features_predict=[]

# 한번에 GPU에 다 안올라가서 쪼개서 넣기
for i in range(0,Number_of_features+1):
    features_predict.extend(front_layers.predict(features[i*3000:(i*3000)+3000], verbose=1))

features_predict=np.array(features_predict)
print("---- features_predict shape")
print(features_predict.shape)

print("---- 400000~800000 training save-----")
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/train_features_800000",features_predict)
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/train_label_800000",label)

features = None
del features
label = None
del label
features_predict=None
del features_predict
gc.collect()

############# test 시작!!!!!!!!
features=np.load("../extracted_feature/whole_shuffle/testing_features.npy")
label=np.load("../extracted_feature/whole_shuffle/testing_label.npy")

print("features shape")
print(features.shape)
print("label shape")
print(label.shape)

Number_of_features=len(features)/3000
Number_of_features=int(Number_of_features)
features_predict=[]

# 한번에 GPU에 다 안올라가서 쪼개서 넣기

for i in range(0, Number_of_features+1):
    features_predict.extend(front_layers.predict(features[i*3000:i*3000+3000], verbose=1))

features_predict=np.array(features_predict)

print("-----features_predict shape-----")
print(features_predict.shape)


print("----- testing save ------")
print("save")
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/testing_features.npy",features_predict)
np.save("/media/2/Network/extracted_feature/whole_shuffle_to_19/testing_label.npy",label)

features=None
del features
label = None
del label
features_predict=None
del features_predict
gc.collect()
