import os,gc
from tqdm import tqdm
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras import layers
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
from keras.models import load_model
from keras.backend import clear_session
#Optimizer
sgd_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

#Loading Datas
datas = np.load("../datasets/testing_features.npy")
label = np.load("../datasets/testing_label.npy")
print("Test data is ready")

#Loading Original Model
original_back_layers = load_model("../original_models/original_back_layers_model.h5")
original_back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
print("Original Model is ready")

#Original accuracy with no feature_map error
print("Original accuracy")
original_back_layers.evaluate(datas,label)
gc.collect()

f = open("total_featuremap.txt","w")
#Testing Retrained models
for Packet_index in range(0,64):
    print("=========================[",Packet_index+1,"]th Packet Error================")
    #Error insertiong
    test_data_with_error = datas.copy()
    test_data_with_error[:,:,:,(Packet_index*8):(Packet_index*8+8)] = 0
    print("Error in Packet [",Packet_index+1,"] is injected")

    gc.collect()

    #Original model with Packet error
    print("original back_layer with [",Packet_index+1,"]th Packet error")
    loss, acc = original_back_layers.evaluate(test_data_with_error,label)
    data = "%f|" % acc
    f.write(data)

    gc.collect()

    #Loading Retrained Model
    for Epoch in range(1,21):
        file_name = "../Back_layers_training_codes/total_feature_map_error_weights/"+str(Epoch)+"Epoch.h5"
        print("load ",file_name)
        retrained_back_layers = load_model(file_name)
        gc.collect()

        #Retrained model with feature_map error
        print("retrained back_layer with [",Packet_index+1,"]th Packet error trained with [",Epoch,"] Epoches")
        loss, acc = retrained_back_layers.evaluate(test_data_with_error,label)
        data = "%f|" % acc
        f.write(data)
        del retrained_back_layers
        clear_session()
    f.write("\n")

    #Freeing Memory
    test_data_with_error = None
    del test_data_with_error
    gc.collect()
    clear_session()
    print("\n\n")

f.close()
