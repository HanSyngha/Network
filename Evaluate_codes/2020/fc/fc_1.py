import os,gc,time
import sys
from tqdm import tqdm
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np
from keras import layers
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
from keras import layers
from keras.models import load_model
from keras.backend import clear_session



#Optimizer
sgd_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
feature_map_per_packet = 8
Packet_number_to_erase = 1
#Loading Datas
label = np.load("/media/0/Network/VGG16/datasets/testing_label.npy")
print("Test data is ready")

#Loading Original Model
original_back_layers = load_model("/media/0/Network/VGG16/pretrained_model/fc123_model.h5")
original_back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
print("Original Model is ready")

gc.collect()


#f = open("../Evaluate_result/"+str(Packet_number_to_erase)+"Single_error_result.txt","w")
#Testing Retrained models
for Packet_index in range(0,64):
#    f.write("\n")

    print("=====================Evaluate [",Packet_index+1,"]th Packet Error================")

    #Error insertiong
    test_data_with_error = np.load("/media/0/Network/VGG16/datasets/testing_features.npy")
    test_data_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
    print("Error in Packet [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"] is injected")
    gc.collect()

    print("Original with error")
    loss, acc = original_back_layers.evaluate(test_data_with_error,label)
    data = "%f|" % acc
    #f.write(data)

    #Loading Retrained Model
    for Epoch in range(1,11):
        clear_session()
        file_name = "/media/1/Network/VGG16/weights/fc/1/"+str(Packet_index+1)+"_"+str(Epoch)+".h5"
        while(os.path.exists(file_name) == False):
            time.sleep(300)
        print("load ",file_name)
        retrained_back_layers = load_model(file_name)
        loss, acc = retrained_back_layers.evaluate(test_data_with_error,label)
        data = "%f|" % acc
        #f.write(data)
        clear_session()
        retrained_back_layers = None
        del retrained_back_layers
        file_name = None
        del file_name
        gc.collect()
        


    #Freeing Memory
    del test_data_with_error
    gc.collect()
    print("\n\n")
    clear_session()
f.close()


