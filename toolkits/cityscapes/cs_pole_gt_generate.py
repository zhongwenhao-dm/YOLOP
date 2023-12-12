import cv2
import os
import json
import numpy as np

# 用来从cityscapes数据集中获得杆状物的label，然后生成杆状物的groundtruth

mode = 'train'
# 分割图真值的路径
root_dir = '/media/user/MyPassport/data/cityscapes/'
img_label_dir = root_dir + 'gtFine_trainvaltest/gtFine/' + mode + '/'
output_gt_dir = root_dir + 'pole_seg_gt/gtFine/' + mode + '/'

# 读取文件夹内的文件夹  
# 图片的路径如下形式：val/城市/图片.png  图片名：munster_000000_000019_gtFine_labelIds.png
son_dir_names = os.listdir(img_label_dir)
for dir_name in son_dir_names:
    city_dir_name = img_label_dir+dir_name+'/'
    file_names_all = os.listdir(city_dir_name)
    file_names = [x for x in file_names_all if x.__contains__('gtFine_labelIds')]
    # print('in '+dir_name)
    # print(len(file_names))

    save_path_dir = output_gt_dir + dir_name + '/'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    # 处理生成对应的gt
    for file_name in file_names:
        img_label = cv2.imread(city_dir_name+file_name, cv2.IMREAD_GRAYSCALE)

        mask = np.where(img_label==17,255,0)
        cv2.imwrite(save_path_dir+file_name, mask) 

