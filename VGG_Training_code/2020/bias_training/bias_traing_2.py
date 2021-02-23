import random
import os,gc,copy,time
import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import Model
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
import math
from keras.backend import clear_session
from keras.models import load_model
from keras import activations

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
tf.random.set_seed(960312)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

class Simple_add(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Simple_add, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.Simple_add = self.add_weight('Simple_add',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.Simple_add


#Global parameters
feature_map_per_packet = 8
Packet_number_to_erase = 1

#Get Training Data
while os.path.exists("/media/0/Network/VGG16/datasets/training_features.npy") == False:
    time.sleep(100)
while os.path.exists("/media/0/Network/VGG16/datasets/training_label.npy") == False:
    time.sleep(100)
while os.path.exists("/media/0/Network/VGG16/datasets/validation_features.npy") == False:
    time.sleep(100)
while os.path.exists("/media/0/Network/VGG16/datasets/validation_label.npy") == False:
    time.sleep(100)

training_feature = np.load("/media/0/Network/VGG16/datasets/training_features.npy")
print("Training Feature",training_feature.shape)
training_label = np.load("/media/0/Network/VGG16/datasets/training_label.npy")
print("Training Label",training_label.shape)
validation_feature = np.load("/media/0/Network/VGG16/datasets/validation_features.npy")
print("Validation Feature",validation_feature.shape)
validation_label = np.load("/media/0/Network/VGG16/datasets/validation_label.npy")
print("Validation Label",validation_label.shape)
print("Datas are Ready")



for Packet_index in range(0,(65-Packet_number_to_erase)):
    print("=============== [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"] Packet error processing ================")
    path = "/media/1/Network/VGG16/weights/bais/2/"+str(Packet_index+1)+"_100.h5"
    if os.path.exists(path):
        print("next")
        continue
        path = None
        del path
        gc.collect()

    del path

    #training_feature_with_error = np.load("/media/0/Network/VGG16/datasets/training_features.npy")
    training_feature_with_error = training_feature.copy()
    #validation_feature_with_error = np.load("/media/0/Network/VGG16/datasets/validation_features.npy")
    validation_feature_with_error = validation_feature.copy()

    print("Insert Error to training set")
    training_feature_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
    print("Error Inserted")
    gc.collect()
  
    print("Insert Error to validation set")
    validation_feature_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
    print("Error Inserted")
    gc.collect()

    print("Initializing Model")

    with strategy.scope():
        sgd_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        input_layer = tf.keras.Input(shape=[4096])
        x  = Simple_add()(input_layer)
        add_layer = tf.keras.Model(inputs=[input_layer], outputs=[x])

        back_layer_original = load_model("/media/0/Network/VGG16/pretrained_model/fc123_model.h5")
        back_layer_original.load_weights("/media/0/Network/VGG16/pretrained_model/fc123_weights.h5")
        back_layer_original.trainable = False
        back_layers = Sequential()
        for idx in range(0,3):
            back_layers.add(back_layer_original.layers[idx])
        back_layers.layers[2].activation = None
        back_layers.add(add_layer)
        back_layers.add(layers.Activation(activations.relu))
        back_layers.add(back_layer_original.layers[3])
        back_layers.build([None,7,7,512])
        back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
        gc.collect()
        back_layers.summary()
  
    print("Training start")
    for Epoch in range(1,11):
        print("[ ",Packet_index+1,":",Packet_index+Packet_number_to_erase," ] Packet error with [ ",Epoch," ] Epochs training")
        file_name = "/media/1/Network/VGG16/weights/bais/2/"+str(Packet_index+1)+"_"+str(Epoch * 10)+".h5"
        if os.path.exists(file_name):
            file_name = None
            del file_name
            print("exsist!")
            continue
    
        back_layers.fit(training_feature_with_error,training_label,batch_size=256,epochs=10,validation_data = (validation_feature_with_error,validation_label),validation_batch_size=256)
        gc.collect()
        print("saving [",file_name,"]")
        back_layers.save(file_name)
        file_name = None
        del file_name
        gc.collect()


    print("Training Finish!")
  
    print("Freeing Datasets")
    clear_session()
    gc.collect()
    training_feature_with_error = None
    del training_feature_with_error
    validation_feature_with_error = None
    del validation_feature_with_error
    back_layers = None
    del back_layers
    back_layer_original = None
    del back_layer_original
    temp = None
    del temp
    clear_session()
    gc.collect()
  



printf("End!!!")
  
