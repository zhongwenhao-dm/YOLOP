import cv2
import os
import json
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import open3d
from sklearn.linear_model import RANSACRegressor

# {
#     "extrinsic": {
#         "baseline": 0.209313, 
#         "pitch": 0.038, 
#         "roll": 0.0, 
#         "x": 1.7, 
#         "y": 0.1, 
#         "yaw": -0.0195, 
#         "z": 1.22
#     }, 
#     "intrinsic": {
#         "fx": 2262.52, 
#         "fy": 2265.3017905988554, 
#         "u0": 1096.98, 
#         "v0": 513.137
#     }
# }

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]

# 对任意p>0, d = ( float(p) - 1. ) / 256， p=0是无效深度
# depth = baseline * fx / disparity

def ransac_plane_sklearn(p3d):
    coe = []

    p3d = np.array(p3d)
    X, y = p3d[:, :2], p3d[:, 2]

    reg=RANSACRegressor(random_state = 0).fit(X,y)
    reg.score(X,y)
    # y_pre = reg.predict(X)
    # y_pre = y_pre.reshape([y_pre.shape[0],1])
    # import pdb; pdb.set_trace()

    # p3d_new = np.concatenate((X,y_pre),axis=1)
    # point_visualization(p3d_new)
    # point_visualization(p3d, p3d_new)

    # y_loss = y_pre - y

    return reg.estimator_.coef_

def ransac_plane_open3d(p3d):
    p3d = np.array(p3d)

    pcd = open3d.geometry.PointCloud()  # 定义点云
    pcd.points = open3d.utility.Vector3dVector(p3d)  # 定义点云坐标位置

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)


    [a, b, c, d] = plane_model

    return [a,b,c,d]


def point_visualization(p3d, p3d_new=None):
    p3d = np.array(p3d)

    test_pcd1 = open3d.geometry.PointCloud()  # 定义点云
    test_pcd1.points = open3d.utility.Vector3dVector(p3d_new)  # 定义点云坐标位置
    # radius=0.01 # 搜索半径
    # max_nn=30 # 邻域内用于估算法线的最大点数
    # # 执行KD树搜索
    # test_pcd1.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius,max_nn))
   
    test_pcd = open3d.geometry.PointCloud()  # 定义点云
    test_pcd.points = open3d.utility.Vector3dVector(p3d)  # 定义点云坐标位置
    open3d.visualization.draw_geometries([test_pcd]+[test_pcd1], window_name="Open3D2", point_show_normal=True)

def projection(p2d, extrin, intrin, disparity):
    p3d = []

    baseline = extrin['baseline']
    fx, fy, cx, cy = intrin['fx'], intrin['fy'], intrin['u0'], intrin['v0']

    for [i, j] in p2d:
        dis = disparity[i, j]
        if dis<=1:
            continue
        dis = (float(dis)-1.0) / 256
        depth = baseline * fx / dis
        # import pdb; pdb.set_trace()

        z = depth
        x = z*(j-cx) / fx
        y = z*(i-cy) / fy

        p3d.append([x,y,z])

    return p3d


if __name__ == '__main__':

    root_dir = '/media/user/MyPassport/data/cityscapes/'
    img_dir = 'gtFine_trainvaltest/gtFine/val/munster/'
    camera_dir = 'camera_trainvaltest/camera/val/munster/'
    img_dis_dir = 'disparity_trainvaltest/disparity/val/munster/'

    img_label_path = root_dir + img_dir + 'munster_000000_000019_gtFine_labelIds.png'
    camera_para_path = root_dir + camera_dir + 'munster_000000_000019_camera.json'
    img_dis_path = root_dir + img_dis_dir + 'munster_000000_000019_disparity.png'

    img_label = cv2.imread(img_label_path, cv2.IMREAD_GRAYSCALE)
    # img_dis = cv2.imread(img_dis_path, cv2.IMREAD_GRAYSCALE)
    img_dis = cv2.imread(img_dis_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    camera_para = json.load(open(camera_para_path))

    # 相机的内外参
    extrinsic = camera_para['extrinsic']
    intrinsic = camera_para['intrinsic']
    

    # 获得2d图片上, label为ground的点的坐标
    h = img_label.shape[0]
    w = img_label.shape[1]
    p_2d = []
    for i in range(h):
        for j in range(w):
            if img_label[i,j]==7:
                p_2d.append([i,j])


    # import pdb; pdb.set_trace()
    # 利用深度图和内参，把2d地面点恢复成3d相机坐标系上的点
    p_3d = projection(p_2d, extrinsic, intrinsic, img_dis)
    p_3d = [p for p in p_3d if p[2]<30]

    # 可视化
    # point_visualization(p_3d, [])

    # ransac算法估计出地面点的平面参数
    # coe = ransac_plane_sklearn(p_3d)
    coe_open3d = ransac_plane_open3d(p_3d)
    # print(y_loss)
    # print(coe)
    print(coe_open3d)

    # 计算得到地平面的法向量




