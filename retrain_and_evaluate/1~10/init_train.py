from random import *
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
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
import time
import datetime 

tf.random.set_seed(960312)


number_of_packet_to_drop = int(sys.argv[1])
drop_num = int(sys.argv[1])
epochs = 100
# {drop_num} packet training input
path = "/media/0/Network/extracted_feature/whole_shuffle_to_19/with_"+str(drop_num)+"_packet_error/"
training_result_file =  "./result/"+str(sys.argv[0])+"_"+str(drop_num)+".txt"
back_layer_file = "/media/2/Network/pretrained_model/back_layers_19~.h5"
model_save_file = "_epoch_"+str(drop_num)+str(sys.argv[0])+".h5"

#val_lr = [1e-4,5e-05]
val_lr = [1e-2]
val_decay = [1e-6,1e-9]
#val_amsgrad = [True,False]
val_amsgrad = [False,True]

f = open(training_result_file,"w")
f.close()
f = open(training_result_file,"a")
print("Adam, pooling5 1 layer training of {} packet_drop(PER = {}/64)".format(drop_num,drop_num),file=f)
f.close()
"""
while True:
    if os.path.exists(path+"test_label_"+str(drop_num)+".npy") == False:
        time.sleep(30*60) # 30 * 60sec => 30 * 1min => 30min
    else :
        f = open(training_result_file,"a")
        print(datetime.datetime.now(),"started",file=f)
        f.close()
        break
"""
train_data = np.load(path+"train_feature_"+str(drop_num)+".npy",mmap_mode="r")
train_label = np.load(path+"train_label_"+str(drop_num)+".npy",mmap_mode="r")
test_data = np.load(path+"test_feature_"+str(drop_num)+".npy",mmap_mode="r")
test_label = np.load(path+"test_label_"+str(drop_num)+".npy",mmap_mode="r")
print("np loading finish")


def packet_drop(arr):
    index = randrange(65-number_of_packet_to_drop)
    arr[:,:,index*8:(index*8+8*number_of_packet_to_drop)] = 0
def error_injection(data):
    for img in tqdm(data):
        packet_drop(img)

def training(lr,decay=0,amsgrad=False):
    f = open(training_result_file,"a")
    print("Adam option : lr: {}, decay: {}, amsgrad: {}".format(lr,decay,amsgrad),file=f)
    f.close()
#    back_layer = load_model(back_layer_file)
    model = VGG16(weights=None,input_shape=(224,224,3))
    back_layer = Sequential([layer for layer in model.layers[19:]])
#    back_layer.layers[1].trainable = True
#    for i in range(2,4):
#        back_layer.layers[i].trainable = False
    
    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9,beta_2=0.999,epsilon=None, decay=decay,amsgrad=amsgrad)
    back_layer.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#    back_layer.build((224,224,3))
#    back_layer.summary()
    f = open(training_result_file,"a")
    test_result =back_layer.evaluate(test_data,test_label)
    print("initial evaluation(not training) == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100),file=f)
    f.close()
    for i in range(epochs):
        back_layer.fit(train_data,train_label, validation_split=0.2, epochs=1, verbose=1,batch_size=256)

        #f = open("result_back_1_adadelta.txt","w")
        test_result =back_layer.evaluate(test_data,test_label)
        f = open(training_result_file,"a")
        print(i,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100),file=f)
        f.close()
        print(i,"th result == loss :{:.4}, Acc : {:.4}% ".format(test_result[0], test_result[1]*100))
        if i %10 == 9:
#            model_save_file = "_epoch_"+str(drop_num)+"_packet_drop_layer1_training.h5"
            model_save = str(i)+model_save_file
            back_layer.save("/media/3/Network/retrain_model_dir/pooling5/1_layer/"+model_save)
        gc.collect()

#f = open("result_back_1_adadelta.txt","w")

gc.collect()
"""
val_lr = [5e-05]
val_decay = [1e-3, 1e-6]
val_amsgrad = [False,True]
"""
for lr in val_lr:
    for decay in val_decay:
        for amsgrad in val_amsgrad:
            print("Adam option : lr: {}, decay: {}, amsgrad: {}".format(lr,decay,amsgrad))
            training(lr,decay,amsgrad)
