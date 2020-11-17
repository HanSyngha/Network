#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1234)
np.random.seed(1234)

############################## clean random set
np_image = np.load("imgnet_val_clean_random_1000.npy", allow_pickle=True)
np_label = np.load("imgnet_val_clean_random_1000_label.npy", allow_pickle=True)
############################## clean random set
random_subset = np_image[:100]
random_subset_label = np_label[:100]

############################## input numpy preprocess in here you can use "random_subset" numpy array

############################## input numpy preprocess in here 

##################################################################################################################################

def gen():
    for x, y in zip(np_image_iter, np_label_iter):
        print("gen x.shape, y.shape", x.shape, y.shape)
        yield(x, y)

def resize(x, y):
    print("shape of x :", x.shape)
    return (tf.image.resize(x, (224,224)), y)

def preprocess(x, y):
    x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    x = tf.cast(x, tf.float32)
    return x, y



def neterror(target, num, error_rate):
    batch, height, width, ch = target.shape
    ret = []
    packet_size = 375
    
    for j in range(batch):
        random_num = []
        temp = tf.reshape(target[j], shape=[-1]).numpy()
        l = len(temp) // packet_size
        if len(temp) % packet_size != 0:
            l += 1
        
        if num == 0:
            packet_error_num = (error_rate * l) // 100
        else:
            packet_error_num = 1

        for i in range(packet_error_num):
            if num <= 0:
                while(1):
                    pindex = random.randint(0, l-1)
                    if pindex not in random_num:
                        random_num.append(pindex)
                        break
            else:
                pindex = num-1

            if pindex == l-1:
                temp[packet_size*pindex : ] = 0
            else:
                temp[packet_size*pindex : packet_size*(pindex+1)] = 0

        ret = np.append(ret, temp, axis=0)

    ret = tf.reshape(ret, shape = [batch, height, width, ch])
    ret = np.array(ret).astype("float32")

    return ret


def compressed_network(compressed):
#def compressed_network(compressed, PER):
    packet_unit_size = 1450
    packet_error_rate = 0.01
#    packet_error_rate = PER

    temp = compressed
    tmpp_f = compressed

    flatten_length = 0
    index_num=[]
    index_num_packet=[]
    packet_flatten_length=0

    la = bytearray(temp[0].numpy())
    ret = []

    for i in range(len(temp)):
        nparr = temp[i].numpy()
        bytarr = bytearray(nparr)
        flatten_length += len(bytarr)*8
        index_num.append(len(bytarr)*8)

        index_num_packet.append(len(bytarr))
        packet_flatten_length += len(bytarr)
        
        if i != 0:
            la += nparr

    for step in range(1):
        cp = la.copy()
        packet_num = int(packet_flatten_length / packet_unit_size) + 1

        packet_error_num = round(packet_error_rate * packet_num)
        random_num = []

        while(packet_error_num > 0):
            while(1):
                rr = random.randint(0, packet_num)
                if rr not in random_num:
                    random_num.append(rr)
                    break
            packet_error_num -= 1

#        random.seed(33)
#        random_sample = random.sample(range(0, packet_num), round(packet_num * packet_error_rate))

        zero_list = [0 for i in range(packet_unit_size)]
#        for index in random_sample:
        for index in random_num:
            start = index*packet_unit_size
            end = start+packet_unit_size
            if end > len(cp)-1:
                end = len(cp)-1
            cp[start:end] = zero_list

        start = 0
        end = 0
        for i in index_num:
            end = end+int(i/8)
            ret.append(tf.convert_to_tensor(np.array(bytes(cp[start:end]))))
            start=end

    ret = np.array(ret)
    return ret

def compress(target, QF, check):
#def compress(target, QF, check, PER):
    batch, height, width, ch = target.shape
    binary_arr = []

    minimum = []
    maximum = []
    decode_feature = []

    original_size = batch * height * width * ch * 4
    compressed_size = 0

    # check == 1 or check == 4 -> quilting channels
    if check == 1 or check == 4:
        ch_ = (height * width * ch) // (224 * 224)

        if ch_ < 1:
            height_ = 112
            width_ = 112
            ch_ = 2
            ll = 112 // height
        else:
            height_ = 224
            width_ = 224
            ll = 224 // height

#        for j in range(batch):
#            target_encode = target[j]


########################################### quilting start
        quil = []
        for k0 in range(ch_):
            for k1 in range(height_):
                for k2 in range(width_):
                    t0 = target[:, k1//ll:k1//ll+1, k2//ll:k2//ll+1, k0*ll*ll+k2%ll+(k1%ll)*ll:k0*ll*ll+k2%ll+(k1%ll)*ll+1]
                    if k2 == 0:
                        t1 = t0
                    else:
                        t1 = np.append(t1, t0, axis=2)
                if k1 == 0:
                    t2 = t1
                else:
                    t2 = np.append(t2, t1, axis=1)
            if k0 == 0:
                quil = t2
            else:
                quil = np.append(quil, t2, axis=-1)

        target = quil.copy()

        for j in range(batch):
            target_encode = tf.reshape(target[j], shape=[-1,ch_])
#            target_encode = tf.reshape(quil, shape = [-1, ch_])

###################### quilting end

            max_ = tf.keras.backend.max(target_encode, axis=0, keepdims = False)
            min_ = tf.keras.backend.min(target_encode, axis=0, keepdims = False)
            target_encode = 255 * ((target_encode - min_) / (max_ - min_))

            target_encode = tf.reshape(target_encode, shape=[height_, width_, ch_]).numpy()
            target_encode = tf.cast(target_encode, dtype=tf.uint8)

            maximum = np.append(maximum, max_, axis=0)
            minimum = np.append(minimum, min_, axis=0)

            feature_encode = [tf.io.encode_jpeg(target_encode[:, :, i:i+1], format='', quality=QF) for i in range(0, ch_)]
            binary_arr = np.append(binary_arr, feature_encode, axis=0)

#        print("encoding end")

#        print("binary_arr.shape :", binary_arr.shape)

        for i in range(batch*ch_):
            compressed_size += len(bytearray(binary_arr[i].numpy()))

#        size_origin = height * width * ch * 4
#        for j in range(batch):
#            temp = 0
#            for i in range(ch_):
#                temp += len(bytearray(binary_arr[ch_*j+i].numpy()))
#            print("Image", j+1, "compression_rate :", round(((size_origin-temp)/size_origin)*100, 2))

        ########### compress end
        # check == 1 -> network_error (X)   /   check == 4 -> network_error (O)
        if check == 1:
            decode_feature = [ np.concatenate( [tf.io.decode_jpeg(binary_arr[i]) for i in range(ch_*j, ch_*(j+1))], axis=-1 ) for j in range(batch) ]
            maximum = tf.reshape(maximum, shape = [batch, ch_]).numpy()
            minumum = tf.reshape(minimum, shape = [batch, ch_]).numpy()
            for j in range(batch):
                decode_feature[j] = tf.reshape(decode_feature[j], shape=[-1, ch_])
                decode_feature[j] = (decode_feature[j]/255) * (maximum[j] - minimum[j]) + minimum[j]
#                decode_feature[j] = tf.reshape(decode_feature[j], shape = [height, width, ch]).numpy()
                decode_feature[j] = tf.reshape(decode_feature[j], shape = [height_, width_, ch_]).numpy()


############################################# origin start

            decode_feature = np.array(decode_feature)

            origin = []
            for k0 in range(ch):
                for k1 in range(height):
                    for k2 in range(width):
#                        t0 = decode_feature[:, (k0%(ll*ll))//ll+k1*ll:(k0%(ll*ll))//ll+k1*ll+1, k0%ll+k2*ll:k0%ll+k2*ll+1, k0//(ll*ll):k0//(ll*ll)+1]
                        t0 = decode_feature[:, (k0%(ll*ll))//ll+k1*ll:(k0%(ll*ll))//ll+k1*ll+1, k0%ll+k2*ll:k0%ll+k2*ll+1, k0//(ll*ll):k0//(ll*ll)+1]

                        if k2 == 0:
                            t1 = t0
                        else:
                            t1 = np.append(t1, t0, axis=2)
                    if k1 == 0:
                        t2 = t1
                    else:
                        t2 = np.append(t2, t1, axis=1)
                if k0 == 0:
                    origin = t2
                else:
                    origin = np.append(origin, t2, axis=-1)

####################### origin end

            decode_feature = origin.copy()

        else:
            temp = []
            decode_temp = []
            binary_arr = compressed_network(binary_arr)
#            binary_arr = compressed_network(binary_arr, (PER/100))

            maximum = tf.reshape(maximum, shape = [batch, ch_]).numpy()
            minimum = tf.reshape(minimum, shape = [batch, ch_]).numpy()

            for j in range(batch):
                for i in range(ch_*j, ch_*(j+1)):
                    try:
                        decode_temp = tf.io.decode_jpeg(binary_arr[i])
                        if decode_temp.shape[0] != height_ or decode_temp.shape[1] != width_:
                            decode_temp = np.zeros((height_, width_, 1))
                    except:
                        decode_temp = np.zeros((height_, width_, 1))
                        
                    if i == ch_*j:
                        temp = decode_temp
                    else:
                        temp = np.concatenate((temp, decode_temp), axis=-1)
                
                temp = tf.reshape(temp, shape=[-1, ch_])
                temp = (temp/255) * (maximum[j] - minimum[j]) + minimum[j]
                temp = tf.reshape(temp, shape = [height_, width_, ch_]).numpy()

                temp = np.expand_dims(temp, axis=0)

                if j == 0:
                    decode_feature = temp
                else:
                    decode_feature = np.concatenate((decode_feature, temp), axis=0)

            decode_feature = np.array(decode_feature)
            origin = []
            for k0 in range(ch):
                for k1 in range(height):
                    for k2 in range(width):
                        t0 = decode_feature[:, (k0%(ll*ll))//ll+k1*ll:(k0%(ll*ll))//ll+k1*ll+1, k0%ll+k2*ll:k0%ll+k2*ll+1, k0//(ll*ll):k0//(ll*ll)+1]
                        
                        if k2 == 0:
                            t1 = t0
                        else:
                            t1 = np.append(t1, t0, axis=2)
                    if k1 == 0:
                        t2 = t1
                    else:
                        t2 = np.append(t2, t1, axis=1)
                if k0 == 0:
                    origin = t2
                else:
                    origin = np.append(origin, t2, axis=-1)

            decode_feature = origin.copy()


    # check == 0 or check == 3 -> original channel compression
    elif check == 0 or check == 3:

        for j in range(batch):
            target_encode = tf.reshape(target[j], shape=[-1,ch])

            max_ = tf.keras.backend.max(target_encode, axis=0, keepdims = False)
            min_ = tf.keras.backend.min(target_encode, axis=0, keepdims = False)
            target_encode = 255 * ((target_encode - min_) / (max_ - min_))

            target_encode = tf.reshape(target_encode, shape=[height, width, ch]).numpy()
            target_encode = tf.cast(target_encode, dtype=tf.uint8)

            maximum = np.append(maximum, max_, axis=0)
            minimum = np.append(minimum, min_, axis=0)

            feature_encode = [tf.io.encode_jpeg(target_encode[:, :, i:i+1], format='', quality=QF) for i in range(ch)]
            binary_arr = np.append(binary_arr, feature_encode, axis=0)

        for i in range(batch * ch):
            compressed_size += len(bytearray(binary_arr[i].numpy()))

#        size_origin = height * width * ch * 4
#        for j in range(batch):
#            temp = 0
#            for i in range(ch):
#                temp += len(bytearray(binary_arr[ch*j+i].numpy()))
#            print("Image", j+1, "compression_rate :", round(((size_origin-temp)/size_origin)*100, 2))

        ############ compress end
        # check == 0 -> network_error (X)   /   check == 3 -> network_error (O)
        if check == 0: 
            decode_feature = [ np.concatenate( [tf.io.decode_jpeg(binary_arr[i]) for i in range(ch*j, ch*(j+1))], axis=-1 ) for j in range(batch) ]
            maximum = tf.reshape(maximum, shape = [batch, ch]).numpy()
            minimum = tf.reshape(minimum, shape = [batch, ch]).numpy()

            for i in range(batch):
                decode_feature[i] = tf.reshape(decode_feature[i], shape=[-1,ch])
                decode_feature[i] = (decode_feature[i]/255) * (maximum[i] - minimum[i]) + minimum[i]
                decode_feature[i] = tf.reshape(decode_feature[i], shape = [height, width, ch]).numpy()

        else:
            temp = []
            decode_temp = []
            binary_arr = compressed_network(binary_arr)
            maximum = tf.reshape(maximum, shape = [batch, ch]).numpy()
            minimum = tf.reshape(minimum, shape = [batch, ch]).numpy()
            
            for j in range(batch):
                for i in range(ch*j, ch*(j+1)):
                    try:
                        decode_temp = tf.io.decode_jpeg(binary_arr[i])
                        if decode_temp.shape[0] != height or decode_temp.shape[1] != width:
                            decode_temp = np.zeros((height, width, 1))
                    except:
                        decode_temp = np.zeros((height, width,1))
                    
                    if i == ch*j:
                        temp = decode_temp
                    else:
                        temp = np.concatenate((temp,decode_temp), axis=-1)

                temp = tf.reshape(temp, shape=[-1,ch])
                temp = (temp/255) * (maximum[j] - minimum[j]) + minimum[j]
                temp = tf.reshape(temp, shape = [height, width, ch]).numpy()

                if j == 0:
                    temp = np.expand_dims(temp, axis=0)
                    decode_feature = temp
                else:
                    temp = np.expand_dims(temp, axis=0)
                    decode_feature = np.concatenate((decode_feature, temp), axis=0)

    # check == 2 or check == 3 -> tiling channels
    elif check == 2 or check == 5:
        
        ch_ = (height * width * ch) // (224 * 224)
        
        if ch_ < 1:
            height_ = 112
            width_ = 112
            ch_ = 2
            ll = 112 // height
        else:
            height_ = 224
            width_ = 224
            ll = 224 // height

        tile = []
        for k0 in range(ch_):
            tmp2 = []
            for k1 in range(ll):
                start = k0*ll*ll + k1*ll
                temp = []
                tmp1 = []
                for k2 in range(ll):
                    temp = target[:, :, :, start+k2:start+k2+1]
                    if k2 == 0:
                        tmp1 = temp
                    else:
                        tmp1 = np.append(tmp1, temp, axis=2)
                if k1 == 0:
                    tmp2 = tmp1
                else:
                    tmp2 = np.append(tmp2, tmp1, axis=1)
            if k0 == 0:
                tile = tmp2
            else:
                tile = np.append(tile, tmp2, axis=-1)

        target = tile.copy()

        for j in range(batch):
            reshape_encode = tf.reshape(target[j], shape=[-1, ch_])
            max_ = tf.keras.backend.max(reshape_encode, axis=0, keepdims = False)
            min_ = tf.keras.backend.min(reshape_encode, axis=0, keepdims = False)
            reshape_encode = 255 * ((reshape_encode - min_) / (max_ - min_))
            reshape_encode = tf.reshape(reshape_encode, shape = [height_, width_, ch_]).numpy()
            reshape_encode = tf.cast(reshape_encode, dtype=tf.uint8)

            maximum = np.append(maximum, max_, axis=0)
            minimum = np.append(minimum, min_, axis=0)

            feature_encode = [tf.io.encode_jpeg(reshape_encode[:, :, i:i+1], format='', quality=QF) for i in range(ch_)]
            binary_arr = np.append(binary_arr, feature_encode, axis=0)

        for i in range(batch*ch_):
            compressed_size += len(bytearray(binary_arr[i].numpy()))

#        size_origin = height*batch*ch*4
#        for j in range(batch):
#            temp = 0
#            for i in range(ch_):
#                temp += len(bytearray(binary_arr[ch_*j+i].numpy()))
#            print("Image", j+1, "compression_rate :", round(((size_origin-temp)/size_origin)*100, 2))

        ############ compress end
        # check == 2 -> network_error (X)   /   check == 5 -> network_error (O)
        if check == 2:
            decode_feature = [ np.concatenate( [tf.io.decode_jpeg(binary_arr[i]) for i in range(ch_*j, ch_*(j+1))], axis=-1) for j in range(batch)]
            maximum = tf.reshape(maximum, shape = [batch, ch_]).numpy()
            minimum = tf.reshape(minimum, shape = [batch, ch_]).numpy()
            for j in range(batch):
                decode_feature[j] = tf.reshape(decode_feature[j], shape=[-1,ch_])
                decode_feature[j] = (decode_feature[j]/255) * (maximum[j] - minimum[j]) + minimum[j]
                decode_feature[j] = tf.reshape(decode_feature[j], shape = [height_, width_, ch_]).numpy()

            decode_feature = np.array(decode_feature)

            origin = []
            for k0 in range(ch_):
                for k1 in range(ll):
                    for k2 in range(ll):
                        tmp = np.expand_dims(decode_feature[:, height*k1:height*(k1+1), width*k2:width*(k2+1), k0], axis=-1)
                        if k0 == 0 and k1 == 0 and k2 == 0:
                            origin = tmp
                        else:
                            origin = np.append(origin, tmp, axis=-1)
            decode_feature = origin.copy()

        else:
            temp = []
            decode_temp = []
            binary_arr = compressed_network(binary_arr)
            maximum = tf.reshape(maximum, shape = [batch, ch_]).numpy()
            minimum = tf.reshape(minimum, shape = [batch, ch_]).numpy()

            for j in range(batch):
                for i in range(ch_*j, ch_*(j+1)):
                    try:
                        decode_temp = tf.io.decode_jpeg(binary_arr[i])
                        if decode_temp.shape[0] != height_ or decode_temp.shape[1] != width_:
                            decode_temp = np.zeros((height_, width_, 1))
                    except:
                        decode_temp = np.zeros((height_, width_, 1))

                    if i == ch_*j:
                        temp = decode_temp
                    else:
                        temp = np.concatenate((temp, decode_temp), axis=-1)

                temp = tf.reshape(temp, shape=[-1, ch_])
                temp = (temp/255) * (maximum[j] - minimum[j]) + minimum[j]
                temp = tf.reshape(temp, shape = [height_, width_, ch_]).numpy()

                temp = np.expand_dims(temp, axis=0)

                if j == 0:
                    decode_feature = temp
                else:
                    decode_feature = np.concatenate((decode_feature, temp), axis=0)

            decode_feature = np.array(decode_feature)

            origin = []
            for k0 in range(ch_):
                for k1 in range(ll):
                    for k2 in range(ll):
                        tmp = np.expand_dims(decode_feature[:, height*k1:height*(k1+1), width*k2:width*(k2+1), k0], axis=-1)
                        if k0 == 0 and k1 == 0 and k2 == 0:
                            origin = tmp
                        else:
                            origin = np.append(origin, tmp, axis=-1)
            decode_feature = origin.copy()


    print("ori :", end=' ')
    print(original_size, end=' ')
    print(",compr :", end=' ')
    print(compressed_size, end=' ')
    print("  =>   compress percentage :", round(((original_size - compressed_size) / original_size) * 100, 2))

    decode_feature = np.array(decode_feature).astype("float32")

    return decode_feature

##################################################################################################################################

np_image_iter = iter(random_subset)
np_label_iter = iter(random_subset_label)   

output_shapes= (tf.TensorShape([None,None,3]), tf.TensorShape([]))

ds = tf.data.Dataset.from_generator(lambda: zip(np_image_iter, np_label_iter), (tf.uint8, tf.int64), output_shapes)  

full_model = tf.keras.applications.VGG19()


ds = ds.map(resize).batch(128).map(lambda x, y: (tf.keras.applications.imagenet_utils.preprocess_input(x), y))

##

lay = 22

half_model_1 = tf.keras.Sequential([layer for layer in full_model.layers[:lay]])
half_model_2 = tf.keras.Sequential([layer for layer in full_model.layers[lay:]])

latent_features = half_model_1.predict(ds)
print("latent_features.shape", latent_features.shape)
print("latent_features.size :", latent_features.size)
batch, height, width, ch = latent_features.shape
######################### compress start

half_model_2.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])

print("[bacth height width ch] :", latent_features.shape)
print("layer :", lay)

# state = 0 -> compress
# state = 1 -> network
# state = 2 -> compress+network
state = 0

if state == 0:
    print("==========compress==========")
elif state == 1:
    print("==========network==========")
else:
    print("==========compress+network==========")

if state == 0:
    mode = 0 # mode = 0 -> original, mode = 1 -> quilting, mode = 2-> tiling

    for mode in range(0, 3):
        if mode == 0: 
            print("original channel compression!")
        elif mode == 1:
            print("quilting channels!")
        else:
            print("tiling channels!")

        if lay == 1 or lay == 2 or lay == 3:
            if mode != 0:
                print("shape is already [224, 224, ]")
                continue

        for i in range(1, 20):
            print("QF :", end=' ')
            print(5*i, end=' ')
            print("->", end=' ')
            decode_feature = compress(latent_features, 5*i, mode) # 3번째 인자 0이면 channel별로 안합치고 1이면 합침
            half_model_2.evaluate(decode_feature, random_subset_label)
    
elif state == 1:
    nn = 0
    if nn == 0:
        for j in range(10):
            print("error_rate :", j+1)
            for i in range(10):
                network_features = neterror(latent_features, 0, j+1)
                half_model_2.evaluate(network_features, random_subset_label)

    elif nn == -1:
        for i in range(15):
            network_features = neterror(latent_features, -1, 0)
            half_model_2.evaluate(network_features, random_subset_label)

    elif nn == 1:
        l = height * width * ch // 375
        if height*width*ch % 375 != 0:
            l += 1
        for i in range(l):
            network_features = neterror(latent_features, i+1, 0)
            half_model_2.evaluate(network_features, random_subset_label)

else:
    num = 100
    print("original channel compression!")
    for i in range(num):
        decode_feature = compress(latent_features, 95, 3) # 3번째 인자 0이면 channel별로 안합치고 1이면 합침
        half_model_2.evaluate(decode_feature, random_subset_label)
    
    if lay != 1:
        print("tiling channels!")
        for i in range(num):
            decode_feature = compress(latent_features, 95, 5) # 3번째 인자 0이면 channel별로 안합치고 1이면 합침
            half_model_2.evaluate(decode_feature, random_subset_label)

        print("quilting channels!")
        for i in range(num):
            decode_feature = compress(latent_features, 95, 4)
            half_model_2.evaluate(decode_feature, random_subset_label)

#        for pp in range(1, 11):
#            for i in range(num):
#                if i == 0:
#                    print("==================== ERROR_RATE :", pp, "====================")
#                decode_feature = compress(latent_features, 95, 4, pp)
#                half_model_2.evaluate(decode_feature, random_subset_label)


print("END")
half_model_2.evaluate(latent_features, random_subset_label)

tf.keras.backend.clear_session()

########################## compress end

############################## intermediate numpy preprocess in here you can use "latent_features" numpy array

############################## intermediate numpy preprocess in here 
#half_model_2.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
#half_model_2.evaluate(decode_feature_, random_subset_label)
#half_model_2.evaluate(decode_feature, random_subset_label)
#half_model_2.evaluate(latent_features, random_subset_label)

#tf.keras.backend.clear_session()


# In[ ]:




