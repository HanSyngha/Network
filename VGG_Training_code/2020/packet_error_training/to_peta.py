import random
import os,gc,time


for Packet_index in range(60,65):
    for Epoch in range(1,11):
        file_name = "/media/1/Network/VGG16/weights/fc/2/"+str(Packet_index+1)+"_"+str(Epoch)+".h5"
        while os.path.exists(file_name) == False:
            time.sleep(300)
        print("scp -P 8002 " + file_name + " gkstmdgk2731@peta.skku.edu:/media/1/Network/VGG16/weights/fc/2/")
        os.system("scp -P 8002 " + file_name + "gkstmdgk2731@peta.skku.edu:/media/1/Network/VGG16/weights/fc/2/")
        os.system("1234")
            



