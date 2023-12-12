import os
import cv2
import tqdm
import shutil

mode = 'train'
save_mode = mode

root_dir = '/home/user/ZWH/code/YOLOP-main/gt/'

dir_10k = root_dir + "bdd100k_images_10k/bdd100k/images/10k/{}/".format(mode)
dir_100k = root_dir + "bdd100k_images_100k/bdd100k/images/100k/{}/".format(mode)

LABELROOT = root_dir + 'det_annotations/data2/zwt/bdd/bdd100k/labels/100k'      # the path of det_annotations folder
MASKROOT = root_dir + 'da_seg_annotations/bdd_seg_gt'                # the path of da_seg_annotations folder
LANEROOT = root_dir + 'll_seg_annotations/bdd_lane_gt'                # the path of ll_seg_annotations folder
POLEROOT = root_dir + 'po_seg_annotations/bdd_pole_gt' 

temp_datadir = root_dir + "temp_data/"



list_10k = os.listdir(dir_10k)

num = 0

for imgname in list_10k:
    path = dir_100k + imgname #.replace('.jpg','.png')
    # print(path)
    
    if os.path.exists(path):
        num = num + 1
        # img = cv2.imread(path)
        if num>2500 :
            save_mode = 'val'
        shutil.copy(path, temp_datadir+"image/{}".format(save_mode))
        # cv2.imwrite(temp_datadir+"image/"+imgname, img)
        label_path = LABELROOT + '/{}/'.format(mode) + imgname.replace('.jpg','.json')
        shutil.copy(label_path, temp_datadir+"det/{}".format(save_mode))

        lane_path = LANEROOT + '/{}/'.format(mode) + imgname.replace('.jpg','.png')
        shutil.copy(lane_path, temp_datadir+"lane/{}".format(save_mode))

        mask_path = MASKROOT + '/{}/'.format(mode) + imgname.replace('.jpg','.png')
        shutil.copy(mask_path, temp_datadir+"drivable/{}".format(save_mode))

        pole_path = POLEROOT + '/{}/'.format(mode) + imgname.replace('.jpg','.png')
        shutil.copy(pole_path, temp_datadir+"pole/{}".format(save_mode))



        

print("{} / {}".format(num,len(list_10k)))