"""
각 폴더별로 지정한 개수만큼 파일을 랜덤하게 지울 때 사용하는 코드입니다.
dataset이 너무 클 때 사용하면 좋습니다.
"""
import os
path="/media/2/Network/Imagenet_subset/train" # train 폴더 안에 1000개의 class 폴더들이 있습니다.
directory_list=os.listdir(path)
print(len(directory_list)) # 1000
Number_of_directory=len(directory_list)
for i in range(0, Number_of_directory):
    directory=path+"/"+directory_list[i]
    os.system("cd " + directory + " && find ./ -type f -print0 | sort -zR | tail -zn +401 | xargs -0 rm")
    #os.system("cd " + directory + " && find ./ -type f -print0 | sort -zR | tail -zn +401")
    # 400개만 남기고 다 지웁니다.
