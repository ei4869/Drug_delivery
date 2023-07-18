import os
import glob

# 使用glob模块获取指定文件夹下的所有jpg文件
jpg_files = glob.glob('D:/Data/opencv_project/Train_database/delivery4/data/val/*.jpg')

# 打开一个txt文件并写入jpg文件的路径
with open('delivery4/data/val.txt', 'w') as f:
    for file in jpg_files:
        f.write(file + '\n')
