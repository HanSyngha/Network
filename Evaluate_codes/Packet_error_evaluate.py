import os,gc,time
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras import layers
from keras.applications import VGG16
from keras.optimizers import SGD
from keras import Sequential
from keras import layers
from keras.models import load_model
from keras.backend import clear_session

#Optimizer
sgd_optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
feature_map_per_packet = 8

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
#original_back_layers.evaluate(datas,label)
gc.collect()

#Testing Retrained models
for Packet_number_to_erase in range(1,11):
    f = open("../Evaluate_result/"+str(Packet_number_to_erase)+"Packet_error_result.txt","w")
    for Packet_index in range(0,(65-Packet_number_to_erase)):
        print("=====================Evaluate [",Packet_index+1,"]th Packet Error================")
        f.write("65.8900|")

        #Error insertiong
        test_data_with_error = datas.copy()
        test_data_with_error[:,:,:,(feature_map_per_packet*Packet_index):(feature_map_per_packet*Packet_index)+(feature_map_per_packet*Packet_number_to_erase)] = 0
        print("Error in Packet [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"] is injected")

        gc.collect()

        #Original model with Packet error
        print("original back_layer with [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"]th Packet error")
        loss, acc = original_back_layers.evaluate(test_data_with_error,label)
        data = "%f|" % acc
        f.write(data)
        gc.collect()

        #Loading Retrained Model
        for Epoch in range(1,11):
            file_name = "../Back_layers_weights/"+str(Packet_number_to_erase)+"_packet_error_weights/"+str(Packet_index+1)+"_Packet_error_"+str(Epoch)+".h5"
            print("load ",file_name)
            original_retrained_back_layers = load_model(file_name)
            gc.collect()

            #Retrained model with feature_map error
            print("retrained back_layer with [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"]th Packet error trained with [",Epoch,"] Epoches")
            loss, acc = original_retrained_back_layers.evaluate(test_data_with_error,label)
            data = "%f|" % acc
            f.write(data)
            del original_retrained_back_layers
            clear_session()
       
        #single layer retraining
        for Epoch in range(1,11):
            file_name = "../Back_layers_weights/"+str(Packet_number_to_erase)+"_packet_error_weights/"+str(Packet_index+1)+"_Packet_error_"+str(Epoch)+".h5"
            print("load ",file_name)
            original_retrained_back_layers = load_model(file_name)
            retrained_back_layers = Sequential([layer for layer in original_retrained_back_layers.layers[:2]])
            retrained_back_layers.add(original_back_layers.layers[2])
            retrained_back_layers.add(original_back_layers.layers[3])
            retrained_back_layers.build([None,7,7,512])
            retrained_back_layers.compile(optimizer=sgd_optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
            del original_retrained_back_layers
            gc.collect()

            #Retrained model with feature_map error
            print("One layer retrained back_layer with [",Packet_index+1,":",Packet_index+Packet_number_to_erase,"]th Packet error trained with [",Epoch,"] Epoches")
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



