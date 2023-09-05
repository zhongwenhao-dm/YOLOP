import os
import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
sys.path.append("/home/user/ZWH/code/YOLOP-main")

from lib.core.postprocess import morphological_process, connect_lane
from lib.utils import plot_one_box,show_seg_result
import cv2
import numpy as np



# 读入图片数据
pole_binary_img_path = '/home/user/ZWH/code/YOLOP-main/gt/po_seg_annotations/a054ee22-00000000.png'
pole_binary_img = cv2.imread(pole_binary_img_path, 0)

# 图片处理
# mask_post = connect_lane(pole_binary_img)
#设置卷积核
kernel = np.ones((3,3), np.uint8)
 
#图像腐蚀处理
erosion = cv2.erode(pole_binary_img, kernel)


# 显示
cv2.imshow("aaa", erosion)
cv2.waitKey(0)

