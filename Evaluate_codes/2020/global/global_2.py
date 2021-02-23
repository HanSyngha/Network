import os,gc,time
import sys
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np
from keras import layers
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
from keras import layers
from keras.models import load_model
from keras.backend import clear_session
from keras.preprocessing.image import ImageDataGenerator

#num = sys.argv[1] # 훈련시킬 layer 수
Epoch = 20


#Optimizer
sgd_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
feature_map_per_packet = 8
#Packet_number_to_erase = int(sys.argv[2]) # 지울 packet 수
#Loading Datas
label = np.load("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/testing_label.npy")
print("Test data is ready")

#Loading Original Model
original_back_layers = load_model("/media/2/Network/pretrained_model/fc123_model.h5")
original_back_layers.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
print("Original Model is ready")

gc.collect()

#original_avg = 0
#retrained_avg = 0
#Testing Retrained models
for Packet_number_to_erase in range(1,21):
    original_avg=0
    retrained_avg=0
    for Packet_index in range(0,65 - int(Packet_number_to_erase)):
    #    f.write("\n")

        print("=====================Evaluate [",Packet_index+1,"]th Packet Error================")

        #Error insertiong
        test_data_with_error = np.load("/media/2/Network/extracted_feature/whole_not_shuffle_to_19/testing_features.npy")
        test_data_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
        print("Error in Packet [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"] is injected")
        gc.collect()

        print("Original with error")
        loss, acc = original_back_layers.evaluate(test_data_with_error,label)
        original_avg = original_avg + acc

        #Loading Retrained Model
        clear_session()
        file_name = "/media/3/Network/weights/global/2/"+str(Epoch)+".h5"
        print("load ",file_name)
        retrained_back_layers = load_model(file_name)
        loss, acc = retrained_back_layers.evaluate(test_data_with_error,label)
        retrained_avg = retrained_avg + acc
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
   
    print("final: ",(original_avg/(65-Packet_number_to_erase)), (retrained_avg/(65-Packet_number_to_erase)))
